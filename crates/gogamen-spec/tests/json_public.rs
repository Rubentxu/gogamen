//! Las primitivas JSON canónicas son API pública (GOG-132).

use gogamen_spec::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};

#[test]
fn canonical_json_primitives_are_public_and_stable() {
    let mut out = String::new();
    write_json_string(&mut out, "a\"b\\c\n\u{1}");
    assert_eq!(out, "\"a\\\"b\\\\c\\n\\u0001\"");

    let mut f = String::new();
    write_canonical_f64(&mut f, -0.0).unwrap();
    f.push(',');
    write_canonical_f64(&mut f, 2.0).unwrap();
    f.push(',');
    write_canonical_f64(&mut f, 0.5).unwrap();
    assert_eq!(f, "0.0,2.0,0.5");
    assert!(write_canonical_f64(&mut String::new(), f64::NAN).is_err());

    let v = parse_json(r#"{"a":[1,-2,1.5,null,true],"b":"x"}"#).unwrap();
    let JsonValue::Object(map) = v else {
        panic!("object expected")
    };
    assert_eq!(
        map["a"],
        JsonValue::Array(vec![
            JsonValue::Number(JsonNumber::I64(1)),
            JsonValue::Number(JsonNumber::I64(-2)),
            JsonValue::Number(JsonNumber::F64(1.5)),
            JsonValue::Null,
            JsonValue::Bool(true),
        ])
    );
    assert!(parse_json(r#"{"a":1,"a":2}"#).is_err());
}

// ---- GOG-137: parse_hex4 no debe hacer panic con UTF-8 multibyte ----

#[test]
fn gog_137_multibyte_in_hex_escape_is_error_not_panic() {
    for input in [
        "\"\\u00é9\"",
        "\"\\uéééé\"",
        "\"\\u12€\"",
        "\"\\u1234€\\u€€€€\"",
        "\"\\ud83d\\u00é9\"",
        "\"\\u😀\"",
    ] {
        assert!(parse_json(input).is_err(), "should be Err: {input}");
    }
}

#[test]
fn gog_137_truncated_and_signed_hex_are_errors() {
    for input in [
        "\"\\u",
        "\"\\u1",
        "\"\\u12",
        "\"\\u123",
        "\"\\u123\"",
        "\\u+123",
        "\"\\u+123\"",
        "\"\\u-123\"",
        "\"\\u 123\"",
    ] {
        assert!(parse_json(input).is_err(), "should be Err: {input}");
    }
}

#[test]
fn gog_137_valid_hex_cases_and_surrogates() {
    assert_eq!(
        parse_json("\"\\u00e9\"").unwrap(),
        JsonValue::String("é".into())
    );
    assert_eq!(
        parse_json("\"\\u00E9\"").unwrap(),
        JsonValue::String("é".into())
    );
    assert_eq!(
        parse_json("\"\\uAbCd\"").unwrap(),
        JsonValue::String("\u{abcd}".into())
    );
    assert_eq!(
        parse_json("\"\\u0041x\"").unwrap(),
        JsonValue::String("Ax".into())
    );
    assert_eq!(
        parse_json("\"\\ud83d\\ude00\"").unwrap(),
        JsonValue::String("😀".into())
    );
    assert_eq!(
        parse_json("\"\\uD83D\\uDE00!\"").unwrap(),
        JsonValue::String("😀!".into())
    );
    assert!(parse_json("\"\\ud83d\"").is_err());
    assert!(parse_json("\"\\ude00\"").is_err());
}

#[test]
fn gog_137_mutation_corpus_never_panics_and_matches_reference() {
    // Referencia previa (canónica) para 4 bytes ASCII: from_str_radix sin '+'.
    fn reference(hex: &str) -> Option<u16> {
        if hex.len() == 4 && hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            u16::from_str_radix(hex, 16).ok()
        } else {
            None
        }
    }
    let fillers = ["é", "€", "😀", "\u{7f}", "+", "-", "g", "\"", "\\", " "];
    let bases = [
        "0041", "00e9", "00E9", "abcd", "ABCD", "1234", "d7ff", "e000",
    ];
    for base in bases {
        for pos in 0..4 {
            for filler in fillers {
                let mut chars: Vec<String> = base.chars().map(String::from).collect();
                chars[pos] = filler.to_string();
                let hex: String = chars.concat();
                let doc = format!("\"\\u{hex}\"");
                let got = std::panic::catch_unwind(|| parse_json(&doc))
                    .unwrap_or_else(|_| panic!("panic on {doc:?}"));
                assert!(got.is_err(), "mutated must be Err: {doc:?}");
                // Truncadas en cada longitud
                for cut in 0..doc.len() {
                    if doc.is_char_boundary(cut) {
                        let _ = std::panic::catch_unwind(|| parse_json(&doc[..cut]))
                            .unwrap_or_else(|_| panic!("panic on prefix of {doc:?}"));
                    }
                }
            }
        }
        // Salida válida idéntica a la referencia.
        let doc = format!("\"\\u{base}\"");
        let expected = reference(base).and_then(|c| char::from_u32(u32::from(c)));
        let got = parse_json(&doc).ok();
        assert_eq!(got, expected.map(|c| JsonValue::String(c.to_string())));
    }
    // Exhaustivo ASCII: toda combinación de 4 bytes de {0,9,a,F,g,+} coincide con la referencia.
    let alphabet = ["0", "9", "a", "F", "g", "+", "é"];
    for a in alphabet {
        for b in alphabet {
            for c in alphabet {
                for d in alphabet {
                    let hex = format!("{a}{b}{c}{d}");
                    let doc = format!("\"\\u{hex}\"");
                    let got = parse_json(&doc).ok();
                    let expected = reference(&hex)
                        .and_then(|v| char::from_u32(u32::from(v)))
                        .map(|c| JsonValue::String(c.to_string()));
                    assert_eq!(got, expected, "{doc:?}");
                }
            }
        }
    }
}
