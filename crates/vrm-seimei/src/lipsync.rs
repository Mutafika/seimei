//! Text-driven lip-sync: turn a string into a timed viseme schedule that
//! [`crate::AvatarController`] can play. Approximate by design — synced to text
//! rate, not audio (there is none). Kana maps to its vowel accurately; kanji has
//! no reading available, so it falls back to a neutral "あ" flap; romaji vowels →
//! their vowel, consonants → a brief closure; punctuation / spaces → a pause.

/// A viseme schedule: a list of `(vowel index / duration)`. The index is
/// `Some(0..=4)` into `[Aa, Ih, Ou, Ee, Oh]`, or `None` for a closed mouth. The
/// duration is in seconds. Feed to [`crate::AvatarController::say`].
pub type Visemes = Vec<(Option<usize>, f32)>;

fn is_cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9FFF}').contains(&c) // CJK ideographs (kanji)
}

/// Build a viseme schedule from text. See the module docs for the mapping.
pub fn text_to_visemes(text: &str) -> Visemes {
    const SEG: f32 = 0.09; // one mora
    const PAUSE: f32 = 0.13;
    const A: &str = "あかさたなはまやらわがざだばぱぁゃゎ";
    const I: &str = "いきしちにひみりぎじぢびぴぃゐ";
    const U: &str = "うくすつぬふむゆるぐずづぶぷぅゅ";
    const E: &str = "えけせてねへめれげぜでべぺぇゑ";
    const O: &str = "おこそとのほもよろをごぞどぼぽぉょ";

    let mut out = Vec::new();
    let mut last: Option<usize> = None;
    for raw in text.chars() {
        // Katakana → hiragana so one table covers both.
        let ch = if ('\u{30A1}'..='\u{30F6}').contains(&raw) {
            char::from_u32(raw as u32 - 0x60).unwrap_or(raw)
        } else {
            raw
        };
        let vowel = if A.contains(ch) {
            Some(0)
        } else if I.contains(ch) {
            Some(1)
        } else if U.contains(ch) {
            Some(2)
        } else if E.contains(ch) {
            Some(3)
        } else if O.contains(ch) {
            Some(4)
        } else {
            None
        };
        if let Some(v) = vowel {
            out.push((Some(v), SEG));
            last = Some(v);
            continue;
        }
        match ch {
            'ー' | 'ｰ' => {
                if let Some(v) = last {
                    out.push((Some(v), SEG)); // prolong: hold the last vowel
                }
            }
            'っ' => out.push((None, SEG * 0.6)), // geminate stop
            'ん' => out.push((None, SEG)),       // nasal → ~closed
            'a' | 'A' => { out.push((Some(0), SEG)); last = Some(0); }
            'i' | 'I' => { out.push((Some(1), SEG)); last = Some(1); }
            'u' | 'U' => { out.push((Some(2), SEG)); last = Some(2); }
            'e' | 'E' => { out.push((Some(3), SEG)); last = Some(3); }
            'o' | 'O' => { out.push((Some(4), SEG)); last = Some(4); }
            c if c.is_ascii_alphabetic() => out.push((None, SEG * 0.5)), // consonant
            c if is_cjk(c) => { out.push((Some(0), SEG)); last = Some(0); } // kanji: neutral flap
            c if c.is_whitespace() => out.push((None, PAUSE)),
            c if c.is_ascii_punctuation() || "、。・「」（）！？…".contains(c) => {
                out.push((None, PAUSE))
            }
            _ => {} // combining / control → skip
        }
    }
    out
}
