//! メッシュの頂点とインデックスを、大きなバッファ（ページ）にまとめて持つ（2026-10-31）。
//!
//! 以前はメッシュごとに頂点とインデックスのバッファを持っていたので、形が全部違うメッシュを描くと、1 つごとに
//! 頂点・インデックスのバッファをつなぎ直す命令が出ていた（2.4 万メッシュで 1 フレームの描画命令の大半）。
//! ページにまとめると、同じページのメッシュはつなぎ直さずに `draw_indexed` の範囲（`first_index` と `base_vertex`）
//! だけで描ける。
//!
//! - 各メッシュの頂点の頭は 256 バイト境界（`VERTEX_ALIGN` 頂点ごと）に置く。メッシュの頂点の範囲だけを
//!   ストレージとしてつなぐ（compute で頂点を書き換える）ときの、オフセットの境界の決まりを満たす
//! - ページは足りなくなったら足す（前のページの倍・上限 `MAX_PAGE_BYTES`）。1 つで上限を超えるメッシュはそれ専用の
//!   ページ。消したメッシュの範囲は空きに戻し、隣の空きとつなげる（最初に収まる空きを使う）

use crate::vertex::GpuVertex;

/// `GpuVertex` の大きさ（64 バイト）
const VERTEX_BYTES: u64 = std::mem::size_of::<GpuVertex>() as u64;
/// メッシュの頂点の頭をそろえる頂点数（256 バイト＝ストレージをオフセットつきでつなぐときの境界）
const VERTEX_ALIGN: u32 = (256 / std::mem::size_of::<GpuVertex>()) as u32;
/// 最初のページの大きさ（頂点 1 万 6 千・インデックス 6 万 5 千）
const FIRST_PAGE_VERTICES: u32 = 16 * 1024;
const FIRST_PAGE_INDICES: u32 = 64 * 1024;
/// ページの大きさの上限（頂点・インデックスとも）
const MAX_PAGE_BYTES: u64 = 128 * 1024 * 1024;

/// 範囲の割り当て（最初に収まる空き・空きは頭の順に並べ、隣どうしはつなげる）
#[derive(Debug)]
struct RangeAlloc {
    /// 空き (頭, 長さ)
    free: Vec<(u32, u32)>,
}

impl RangeAlloc {
    fn new(cap: u32) -> Self {
        Self { free: if cap > 0 { vec![(0, cap)] } else { Vec::new() } }
    }

    /// `len` を `align` の倍数の頭で取る。取れなければ None。`len == 0` は 0 を返す（何も取らない）
    fn alloc(&mut self, len: u32, align: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }
        for k in 0..self.free.len() {
            let (s, l) = self.free[k];
            let a = s.div_ceil(align) * align;
            let end = s + l;
            if a + len > end {
                continue;
            }
            let mut pieces = Vec::with_capacity(2);
            if a > s {
                pieces.push((s, a - s));
            }
            if a + len < end {
                pieces.push((a + len, end - a - len));
            }
            self.free.splice(k..=k, pieces);
            return Some(a);
        }
        None
    }

    /// 取った範囲を空きに戻す（隣の空きとつなげる）
    fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        let k = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(k, (start, len));
        // 後ろとつなげる
        if k + 1 < self.free.len() && self.free[k].0 + self.free[k].1 == self.free[k + 1].0 {
            self.free[k].1 += self.free[k + 1].1;
            self.free.remove(k + 1);
        }
        // 前とつなげる
        if k > 0 && self.free[k - 1].0 + self.free[k - 1].1 == self.free[k].0 {
            self.free[k - 1].1 += self.free[k].1;
            self.free.remove(k);
        }
    }
}

/// 1 ページ（頂点とインデックスのバッファの組）
pub(crate) struct Page {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    vertices: RangeAlloc,
    indices: RangeAlloc,
}

/// 1 メッシュに割り当てた範囲（頂点とインデックスは同じページ）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slot {
    pub page: u32,
    /// 頂点の頭（`draw_indexed` の `base_vertex`）と、取った頂点の数
    pub base_vertex: u32,
    pub vertex_cap: u32,
    /// インデックスの頭（`draw_indexed` の範囲の頭）と、取ったインデックスの数
    pub first_index: u32,
    pub index_cap: u32,
}

impl Slot {
    /// 頂点 `vertices`・インデックス `indices` がこの範囲に収まるか（収まれば書き直すだけでよい）
    pub fn fits(&self, vertices: u32, indices: u32) -> bool {
        vertices <= self.vertex_cap && indices <= self.index_cap
    }
}

pub(crate) struct MeshArena {
    pages: Vec<Page>,
    /// 次に作るページの大きさ（頂点・インデックスの数）
    next_vertices: u32,
    next_indices: u32,
}

impl MeshArena {
    pub fn new() -> Self {
        Self { pages: Vec::new(), next_vertices: FIRST_PAGE_VERTICES, next_indices: FIRST_PAGE_INDICES }
    }

    pub fn page(&self, i: u32) -> &Page {
        &self.pages[i as usize]
    }

    /// 頂点 `vertices`・インデックス `indices` の範囲を取る（どのページにも入らなければページを足す）
    pub fn alloc(&mut self, device: &wgpu::Device, vertices: u32, indices: u32) -> Slot {
        for (i, p) in self.pages.iter_mut().enumerate() {
            if let Some(slot) = Self::alloc_in(p, i as u32, vertices, indices) {
                return slot;
            }
        }
        let max_v = (MAX_PAGE_BYTES / VERTEX_BYTES) as u32;
        let max_i = (MAX_PAGE_BYTES / 4) as u32;
        let cap_v = self.next_vertices.max(vertices.div_ceil(VERTEX_ALIGN) * VERTEX_ALIGN);
        let cap_i = self.next_indices.max(indices);
        self.next_vertices = (self.next_vertices.saturating_mul(2)).min(max_v);
        self.next_indices = (self.next_indices.saturating_mul(2)).min(max_i);
        let n = self.pages.len();
        let page = Page {
            vertex_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Mesh Vertex Page {n}")),
                size: cap_v as u64 * VERTEX_BYTES,
                // STORAGE: compute シェーダが頂点を直接書き込めるように（メッシュの範囲だけをつなぐ）
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            index_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Mesh Index Page {n}")),
                size: (cap_i.max(1) as u64) * 4,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            vertices: RangeAlloc::new(cap_v),
            indices: RangeAlloc::new(cap_i),
        };
        self.pages.push(page);
        let last = self.pages.len() - 1;
        Self::alloc_in(&mut self.pages[last], last as u32, vertices, indices).expect("足したページには収まる")
    }

    fn alloc_in(p: &mut Page, page: u32, vertices: u32, indices: u32) -> Option<Slot> {
        let base_vertex = p.vertices.alloc(vertices, VERTEX_ALIGN)?;
        let Some(first_index) = p.indices.alloc(indices, 1) else {
            p.vertices.free(base_vertex, vertices);
            return None;
        };
        Some(Slot { page, base_vertex, vertex_cap: vertices, first_index, index_cap: indices })
    }

    /// 範囲を空きに戻す
    pub fn free(&mut self, slot: &Slot) {
        if let Some(p) = self.pages.get_mut(slot.page as usize) {
            p.vertices.free(slot.base_vertex, slot.vertex_cap);
            p.indices.free(slot.first_index, slot.index_cap);
        }
    }

    /// 範囲に頂点とインデックスを書く（インデックスはメッシュの中の番号のまま。`base_vertex` で足す）
    pub fn write(&self, queue: &wgpu::Queue, slot: &Slot, vertices: &[GpuVertex], indices: &[u32]) {
        let p = self.page(slot.page);
        if !vertices.is_empty() {
            queue.write_buffer(&p.vertex_buffer, slot.base_vertex as u64 * VERTEX_BYTES, bytemuck::cast_slice(vertices));
        }
        if !indices.is_empty() {
            queue.write_buffer(&p.index_buffer, slot.first_index as u64 * 4, bytemuck::cast_slice(indices));
        }
    }

    /// 全部捨てる（ページも返す）
    pub fn clear(&mut self) {
        *self = Self::new();
    }
}

/// 頂点 1 つのバイト数（`MeshInstance` の切り出しで使う）
pub(crate) const fn vertex_bytes() -> u64 {
    VERTEX_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 取った範囲どうしは重ならず、全部戻すと 1 つの空きに戻る（乱数で取る・戻すを繰り返す）
    #[test]
    fn ranges_never_overlap_and_coalesce_back() {
        let cap = 10_000;
        let mut a = RangeAlloc::new(cap);
        let mut taken: Vec<(u32, u32)> = Vec::new();
        let mut seed = 12345u64;
        let mut rnd = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32) % n
        };
        for _ in 0..5000 {
            if taken.is_empty() || rnd(3) > 0 {
                let len = 1 + rnd(200);
                let align = [1, 4][rnd(2) as usize];
                if let Some(s) = a.alloc(len, align) {
                    assert_eq!(s % align, 0, "頭がそろっていない");
                    assert!(s + len <= cap);
                    for &(t, l) in &taken {
                        assert!(s + len <= t || t + l <= s, "重なった: ({s},{len}) と ({t},{l})");
                    }
                    taken.push((s, len));
                }
            } else {
                let k = rnd(taken.len() as u32) as usize;
                let (s, l) = taken.swap_remove(k);
                a.free(s, l);
            }
            // 空きは頭の順で、隣どうしはつながっている（間に取った範囲がある）
            for w in a.free.windows(2) {
                assert!(w[0].0 + w[0].1 < w[1].0, "空きが並んでいない／つながっていない: {:?}", w);
            }
        }
        for (s, l) in taken.drain(..) {
            a.free(s, l);
        }
        assert_eq!(a.free, vec![(0, cap)]);
    }

    #[test]
    fn zero_length_takes_nothing() {
        let mut a = RangeAlloc::new(8);
        assert_eq!(a.alloc(0, 4), Some(0));
        assert_eq!(a.free, vec![(0, 8)]);
        a.free(0, 0);
        assert_eq!(a.free, vec![(0, 8)]);
        assert_eq!(a.alloc(9, 1), None);
    }
}
