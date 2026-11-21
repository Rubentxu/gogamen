//! Parser y serializador JSON estricto en `std` puro.
//!
//! Primitivas canónicas públicas compartidas por spec, compilador y CLI:
//! escape de strings, formato de `f64` (`-0.0` se normaliza a `0.0`, siempre con
//! parte decimal o exponente) y parser estricto (sin claves duplicadas, profundidad máx. 64).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::error::SpecError;

const MAX_JSON_DEPTH: usize = 64;

/// Número JSON preservando enteros exactos.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonNumber {
    I64(i64),
    U64(u64),
    F64(f64),
}

/// Valor JSON parseado (objetos ordenados por clave).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(JsonNumber),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

/// Parsea un documento JSON estricto.
pub fn parse_json(input: &str) -> Result<JsonValue, SpecError> {
    let mut parser = JsonParser::new(input);
    let value = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(SpecError::JsonSyntax {
            offset: parser.pos,
            message: "trailing characters after JSON root value".to_owned(),
        });
    }
    Ok(value)
}

/// Escribe un string JSON escapado canónicamente.
pub fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Escribe un `f64` finito en forma canónica; error si no es finito.
pub fn write_canonical_f64(out: &mut String, value: f64) -> Result<(), SpecError> {
    if !value.is_finite() {
        return Err(SpecError::NonFiniteFloat {
            context: "json serializer".to_owned(),
        });
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    let formatted = format!("{normalized}");
    out.push_str(&formatted);
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        out.push_str(".0");
    }
    Ok(())
}

struct JsonParser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn err(&self, message: impl Into<String>) -> SpecError {
        SpecError::JsonSyntax {
            offset: self.pos,
            message: message.into(),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next_byte(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, SpecError> {
        if depth > MAX_JSON_DEPTH {
            return Err(self.err("maximum JSON nesting depth exceeded"));
        }
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(self.err("unexpected end of input"));
        };
        match b {
            b'n' => self.parse_null(),
            b't' | b'f' => self.parse_bool(),
            b'"' => self.parse_string().map(JsonValue::String),
            b'[' => self.parse_array(depth),
            b'{' => self.parse_object(depth),
            b'-' | b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            other => Err(self.err(format!("unexpected byte 0x{other:02x}"))),
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, SpecError> {
        if self.src[self.pos..].starts_with("null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(self.err("invalid literal, expected 'null'"))
        }
    }

    fn parse_bool(&mut self) -> Result<JsonValue, SpecError> {
        if self.src[self.pos..].starts_with("true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.src[self.pos..].starts_with("false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.err("invalid boolean literal"))
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, SpecError> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            items.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            match self.next_byte() {
                Some(b',') => continue,
                Some(b']') => return Ok(JsonValue::Array(items)),
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, SpecError> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut map = BTreeMap::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expected string key in object"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next_byte() != Some(b':') {
                return Err(self.err("expected ':' after object key"));
            }
            let value = self.parse_value(depth + 1)?;
            if map.insert(key.clone(), value).is_some() {
                return Err(SpecError::DuplicateJsonKey { key });
            }
            self.skip_ws();
            match self.next_byte() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(map)),
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, SpecError> {
        self.pos += 1; // opening '"'
        let mut out = String::new();
        while let Some(b) = self.peek() {
            match b {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self
                        .next_byte()
                        .ok_or_else(|| self.err("unterminated escape"))?;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if !self.src[self.pos..].starts_with("\\u") {
                                    return Err(self.err("unpaired high UTF-16 surrogate"));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(self.err("invalid low UTF-16 surrogate"));
                                }
                                let combined = 0x10000
                                    + (((u32::from(code) - 0xD800) << 10)
                                        | (u32::from(low) - 0xDC00));
                                let ch = char::from_u32(combined)
                                    .ok_or_else(|| self.err("invalid unicode scalar"))?;
                                out.push(ch);
                            } else if (0xDC00..=0xDFFF).contains(&code) {
                                return Err(self.err("unexpected low UTF-16 surrogate"));
                            } else {
                                let ch = char::from_u32(u32::from(code))
                                    .ok_or_else(|| self.err("invalid unicode scalar"))?;
                                out.push(ch);
                            }
                        }
                        _ => return Err(self.err("invalid escape sequence")),
                    }
                }
                c if c < 0x20 => {
                    return Err(self.err("unescaped control character in string"));
                }
                _ => {
                    let ch = self.src[self.pos..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.err("invalid UTF-8"))?;
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
        Err(self.err("unterminated string literal"))
    }

    fn parse_hex4(&mut self) -> Result<u16, SpecError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(self.err("incomplete \\uXXXX escape"));
        }
        let hex = &self.src[self.pos..self.pos + 4];
        self.pos += 4;
        u16::from_str_radix(hex, 16).map_err(|_| self.err("invalid hex in \\uXXXX escape"))
    }

    fn parse_number(&mut self) -> Result<JsonNumber, SpecError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("leading zero in JSON number"));
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(self.err("invalid number")),
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit after decimal point"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.err("expected digit in exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let slice = &self.src[start..self.pos];
        if is_float {
            let v: f64 = slice.parse().map_err(|_| self.err("invalid f64 number"))?;
            if !v.is_finite() {
                return Err(self.err("non-finite float literal"));
            }
            Ok(JsonNumber::F64(if v == 0.0 { 0.0 } else { v }))
        } else if slice.starts_with('-') {
            if let Ok(v) = slice.parse::<i64>() {
                Ok(JsonNumber::I64(v))
            } else {
                let v: f64 = slice.parse().map_err(|_| self.err("invalid integer"))?;
                if !v.is_finite() {
                    return Err(self.err("non-finite float literal"));
                }
                Ok(JsonNumber::F64(v))
            }
        } else if let Ok(v) = slice.parse::<i64>() {
            Ok(JsonNumber::I64(v))
        } else if let Ok(v) = slice.parse::<u64>() {
            Ok(JsonNumber::U64(v))
        } else {
            let v: f64 = slice.parse().map_err(|_| self.err("invalid integer"))?;
            if !v.is_finite() {
                return Err(self.err("non-finite float literal"));
            }
            Ok(JsonNumber::F64(v))
        }
    }
}
