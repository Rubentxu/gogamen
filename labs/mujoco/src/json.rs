//! Minimal JSON value + parser (no external crates) for the sidecar line protocol.
//! Floats are printed with Rust's shortest round-trip format and parsed with `str::parse`,
//! so `f64` values survive the process boundary bit-exactly.

/// JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// null
    Null,
    /// bool
    Bool(bool),
    /// number
    Num(f64),
    /// string
    Str(String),
    /// array
    Arr(Vec<Json>),
    /// object (insertion order)
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Object member lookup.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// As f64.
    pub fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    /// As string.
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    /// As bool.
    pub fn as_bool(&self) -> Option<bool> {
        if let Json::Bool(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    /// As numeric vector.
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Json::Arr(a) => a.iter().map(Json::as_f64).collect(),
            _ => None,
        }
    }
}

/// Format a float slice as a JSON array (`{:?}` is the shortest round-trip form).
pub fn fmt_arr(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
    format!("[{}]", items.join(","))
}

/// Parse a JSON document.
pub fn parse(s: &str) -> Result<Json, String> {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing characters at {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at {}", c as char, self.i))
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(format!("bad literal at {}", self.i))
        }
    }
    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i) {
            None => Err("unexpected end".into()),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'[') => {
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(out));
                }
                loop {
                    out.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(out));
                        }
                        _ => return Err(format!("bad array at {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut out = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(out));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.eat(b':')?;
                    out.push((k, self.value()?));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(out));
                        }
                        _ => return Err(format!("bad object at {}", self.i)),
                    }
                }
            }
            Some(_) => {
                let start = self.i;
                while self.i < self.b.len()
                    && matches!(
                        self.b[self.i],
                        b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                    )
                {
                    self.i += 1;
                }
                std::str::from_utf8(&self.b[start..self.i])
                    .ok()
                    .and_then(|t| t.parse::<f64>().ok())
                    .map(Json::Num)
                    .ok_or_else(|| format!("bad number at {start}"))
            }
        }
    }
    fn string(&mut self) -> Result<String, String> {
        if self.b.get(self.i) != Some(&b'"') {
            return Err(format!("expected string at {}", self.i));
        }
        self.i += 1;
        let mut out = String::new();
        loop {
            match self.b.get(self.i) {
                None => return Err("unterminated string".into()),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    match self.b.get(self.i) {
                        Some(b'n') => out.push('\n'),
                        Some(b't') => out.push('\t'),
                        Some(b'r') => out.push('\r'),
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        _ => return Err("unsupported escape".into()),
                    }
                    self.i += 1;
                }
                Some(_) => {
                    let rest = std::str::from_utf8(&self.b[self.i..]).map_err(|e| e.to_string())?;
                    let ch = rest.chars().next().ok_or("eof")?;
                    out.push(ch);
                    self.i += ch.len_utf8();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_floats_bit_exactly() {
        let xs = [
            0.1,
            -1e-300,
            1.7976931348623157e308,
            123_456.789_012_345_67,
            0.0,
        ];
        let j = parse(&format!(
            "{{\"a\":{},\"s\":\"x\\\"y\",\"b\":true}}",
            fmt_arr(&xs)
        ))
        .unwrap();
        let back = j.get("a").unwrap().as_f64_vec().unwrap();
        for (a, b) in xs.iter().zip(&back) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(j.get("s").unwrap().as_str(), Some("x\"y"));
        assert_eq!(j.get("b").unwrap().as_bool(), Some(true));
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("{\"a\":").is_err());
        assert!(parse("[1,2] x").is_err());
    }
}
