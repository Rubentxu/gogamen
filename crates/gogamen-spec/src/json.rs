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

/// Límite por defecto de bytes totales de un documento JSON (16 MiB).
pub const DEFAULT_JSON_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Límite por defecto de valores JSON (incluye raíz, elementos de arrays y valores de objetos).
pub const DEFAULT_JSON_MAX_ELEMENTS: usize = 1_000_000;
/// Límite por defecto de longitud (bytes UTF-8 decodificados) de un string o clave (1 MiB).
pub const DEFAULT_JSON_MAX_STRING_BYTES: usize = 1024 * 1024;

/// Límites de recursos configurables para [`parse_with_limits`].
///
/// La profundidad sigue fija (64). Los defaults son prudentes y holgados para specs reales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonLimits {
    /// Bytes totales máximos del documento.
    pub max_bytes: usize,
    /// Número máximo de valores JSON parseados.
    pub max_elements: usize,
    /// Longitud máxima en bytes de cada string (claves incluidas), ya decodificado.
    pub max_string_bytes: usize,
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_JSON_MAX_BYTES,
            max_elements: DEFAULT_JSON_MAX_ELEMENTS,
            max_string_bytes: DEFAULT_JSON_MAX_STRING_BYTES,
        }
    }
}

/// Parsea un documento JSON estricto con los [`JsonLimits`] por defecto.
pub fn parse_json(input: &str) -> Result<JsonValue, SpecError> {
    parse_with_limits(input, JsonLimits::default())
}

/// Parsea un documento JSON estricto aplicando límites explícitos de bytes, elementos y strings.
pub fn parse_with_limits(input: &str, limits: JsonLimits) -> Result<JsonValue, SpecError> {
    if input.len() > limits.max_bytes {
        return Err(SpecError::JsonSyntax {
            offset: 0,
            message: format!(
                "JSON input exceeds maximum size of {} bytes",
                limits.max_bytes
            ),
        });
    }
    let mut parser = JsonParser::new(input, limits);
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
    limits: JsonLimits,
    elements: usize,
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a str, limits: JsonLimits) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            limits,
            elements: 0,
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
        self.elements += 1;
        if self.elements > self.limits.max_elements {
            return Err(self.err(format!(
                "JSON value count exceeds maximum of {}",
                self.limits.max_elements
            )));
        }
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
            if out.len() > self.limits.max_string_bytes {
                return Err(self.err(format!(
                    "JSON string exceeds maximum of {} bytes",
                    self.limits.max_string_bytes
                )));
            }
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
        let mut value: u16 = 0;
        for i in 0..4 {
            let digit = match self.bytes[self.pos + i] {
                b @ b'0'..=b'9' => b - b'0',
                b @ b'a'..=b'f' => b - b'a' + 10,
                b @ b'A'..=b'F' => b - b'A' + 10,
                _ => {
                    self.pos += 4;
                    return Err(self.err("invalid hex in \\uXXXX escape"));
                }
            };
            value = (value << 4) | u16::from(digit);
        }
        self.pos += 4;
        Ok(value)
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

#[cfg(test)]
mod limit_tests {
    use super::*;

    fn small(max_bytes: usize, max_elements: usize, max_string_bytes: usize) -> JsonLimits {
        JsonLimits {
            max_bytes,
            max_elements,
            max_string_bytes,
        }
    }

    #[test]
    fn gog_138_default_api_matches_explicit_defaults() {
        let doc = r#"{"a":[1,2,{"b":"x"}],"c":null}"#;
        assert_eq!(
            parse_json(doc).unwrap(),
            parse_with_limits(doc, JsonLimits::default()).unwrap()
        );
    }

    #[test]
    fn gog_138_rejects_oversized_bytes_before_parsing() {
        let err = parse_with_limits("[1,2,3,4]", small(5, 100, 100)).unwrap_err();
        assert!(err.to_string().contains("maximum size"), "{err}");
        assert!(parse_with_limits("[1,2]", small(5, 100, 100)).is_ok());
    }

    #[test]
    fn gog_138_rejects_too_many_elements() {
        let doc = format!("[{}]", vec!["0"; 50].join(","));
        let err = parse_with_limits(&doc, small(10_000, 10, 100)).unwrap_err();
        assert!(err.to_string().contains("value count"), "{err}");
        // raíz + 50 elementos = 51 valores
        assert!(parse_with_limits(&doc, small(10_000, 51, 100)).is_ok());
    }

    #[test]
    fn gog_138_rejects_long_strings_and_keys() {
        let long = "a".repeat(64);
        let s = format!("\"{long}\"");
        assert!(parse_with_limits(&s, small(1000, 10, 63)).is_err());
        assert!(parse_with_limits(&s, small(1000, 10, 64)).is_ok());
        let k = format!("{{\"{long}\":1}}");
        assert!(parse_with_limits(&k, small(1000, 10, 63)).is_err());
        // escapes decodificados cuentan por su longitud final
        let esc = format!("\"{}\"", "\\n".repeat(40));
        assert!(parse_with_limits(&esc, small(1000, 10, 39)).is_err());
        assert!(parse_with_limits(&esc, small(1000, 10, 40)).is_ok());
    }
}
