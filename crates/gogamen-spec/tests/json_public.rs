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
