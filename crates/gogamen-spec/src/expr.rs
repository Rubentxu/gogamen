//! Expresiones simbólicas declarativas tipadas (`std`-only, sin código arbitrario).
//!
//! Este módulo define el AST restringido [`ExprAst`], los tipos [`ExprType`] y las
//! unidades físicas explícitas [`ExprUnit`], junto con su validación estructural y
//! serialización/deserialización JSON canónica.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{SpecError, UnknownFieldPolicy};
use crate::json::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};

/// Profundidad máxima permitida para un [`ExprAst`] en validación y serialización.
pub const EXPR_AST_MAX_DEPTH: usize = 64;

/// Unidad física o dimensional explícita de un valor escalar en [`ExprAst`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExprUnit {
    /// Magnitud adimensional pura.
    Dimensionless,
    /// Energía (por ejemplo, presupuesto o nivel energético de un agente).
    Energy,
    /// Tiempo en segundos.
    Seconds,
    /// Longitud o distancia en metros.
    Meters,
    /// Razón o factor adimensional acotado/relativo (por ejemplo, curiosidad).
    Ratio,
    /// Velocidad en metros por segundo (`Meters / Seconds`).
    MetersPerSecond,
    /// Potencia o tasa energética (`Energy / Seconds`).
    Power,
}

impl ExprUnit {
    /// Representación textual canónica de la unidad para JSON y diagnóstico.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dimensionless => "dimensionless",
            Self::Energy => "energy",
            Self::Seconds => "seconds",
            Self::Meters => "meters",
            Self::Ratio => "ratio",
            Self::MetersPerSecond => "meters_per_second",
            Self::Power => "power",
        }
    }

    /// Parsea una unidad a partir de su representación textual canónica.
    pub fn parse(raw: &str) -> Result<Self, SpecError> {
        match raw {
            "dimensionless" => Ok(Self::Dimensionless),
            "energy" => Ok(Self::Energy),
            "seconds" => Ok(Self::Seconds),
            "meters" => Ok(Self::Meters),
            "ratio" => Ok(Self::Ratio),
            "meters_per_second" => Ok(Self::MetersPerSecond),
            "power" => Ok(Self::Power),
            other => Err(SpecError::InvalidFieldValue {
                context: "ExprUnit",
                field: "unit",
                reason: format!("unknown unit '{other}'"),
            }),
        }
    }

    /// Combina unidades bajo multiplicación (`self * rhs`), o `None` si no hay unidad definida.
    pub const fn combine_mul(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (Self::Dimensionless, u) | (u, Self::Dimensionless) => Some(u),
            (Self::Ratio, u) | (u, Self::Ratio) => Some(u),
            (Self::MetersPerSecond, Self::Seconds) | (Self::Seconds, Self::MetersPerSecond) => {
                Some(Self::Meters)
            }
            (Self::Power, Self::Seconds) | (Self::Seconds, Self::Power) => Some(Self::Energy),
            _ => None,
        }
    }

    /// Combina unidades bajo división (`self / rhs`), o `None` si no hay unidad definida.
    pub const fn combine_div(self, rhs: Self) -> Option<Self> {
        match (self, rhs) {
            (u, Self::Dimensionless) | (u, Self::Ratio) => Some(u),
            (Self::Meters, Self::Seconds) => Some(Self::MetersPerSecond),
            (Self::Meters, Self::MetersPerSecond) => Some(Self::Seconds),
            (Self::Energy, Self::Seconds) => Some(Self::Power),
            (Self::Energy, Self::Power) => Some(Self::Seconds),
            (a, b) if a as u8 == b as u8 => Some(Self::Dimensionless),
            _ => None,
        }
    }
}

impl fmt::Display for ExprUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tipo estático de una expresión o variable simbólica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExprType {
    /// Escalar de punto flotante (`f64` finito) con unidad explícita.
    Scalar(ExprUnit),
    /// Valor booleano.
    Bool,
}

impl fmt::Display for ExprType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(unit) => write!(f, "Scalar({unit})"),
            Self::Bool => f.write_str("Bool"),
        }
    }
}

/// AST restringido de expresiones simbólicas declarativas.
///
/// No admite bucles, llamadas a funciones arbitrarias, IO ni efectos secundarios.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprAst {
    /// Constante escalar `f64` con unidad explícita.
    ConstF64 {
        /// Valor numérico (debe ser finito al validar/compilar).
        value: f64,
        /// Unidad física del literal.
        unit: ExprUnit,
    },
    /// Constante booleana.
    ConstBool(bool),
    /// Referencia a una variable por nombre.
    Var(String),
    /// Suma (`lhs + rhs`).
    Add(Box<ExprAst>, Box<ExprAst>),
    /// Resta (`lhs - rhs`).
    Sub(Box<ExprAst>, Box<ExprAst>),
    /// Multiplicación (`lhs * rhs`).
    Mul(Box<ExprAst>, Box<ExprAst>),
    /// División (`lhs / rhs`).
    Div(Box<ExprAst>, Box<ExprAst>),
    /// Menor estricto (`lhs < rhs`).
    Lt(Box<ExprAst>, Box<ExprAst>),
    /// Menor o igual (`lhs <= rhs`).
    Le(Box<ExprAst>, Box<ExprAst>),
    /// Mayor estricto (`lhs > rhs`).
    Gt(Box<ExprAst>, Box<ExprAst>),
    /// Mayor o igual (`lhs >= rhs`).
    Ge(Box<ExprAst>, Box<ExprAst>),
    /// Igualdad (`lhs == rhs`).
    Eq(Box<ExprAst>, Box<ExprAst>),
    /// Desigualdad (`lhs != rhs`).
    Ne(Box<ExprAst>, Box<ExprAst>),
    /// Conjunción lógica (`lhs && rhs`).
    And(Box<ExprAst>, Box<ExprAst>),
    /// Disyunción lógica (`lhs || rhs`).
    Or(Box<ExprAst>, Box<ExprAst>),
    /// Negación lógica (`!expr`).
    Not(Box<ExprAst>),
}

impl ExprAst {
    /// Construye una constante escalar adimensional verificando que sea finita.
    pub fn const_f64(value: f64) -> Result<Self, SpecError> {
        Self::const_f64_with_unit(value, ExprUnit::Dimensionless)
    }

    /// Construye una constante escalar con unidad explícita verificando que sea finita.
    pub fn const_f64_with_unit(value: f64, unit: ExprUnit) -> Result<Self, SpecError> {
        if !value.is_finite() {
            return Err(SpecError::NonFiniteFloat {
                context: "ExprAst::ConstF64".to_owned(),
            });
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        Ok(Self::ConstF64 {
            value: normalized,
            unit,
        })
    }

    /// Construye una constante booleana.
    pub const fn const_bool(value: bool) -> Self {
        Self::ConstBool(value)
    }

    /// Construye una referencia a variable validando que el nombre no esté vacío.
    pub fn var(name: impl Into<String>) -> Result<Self, SpecError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SpecError::EmptyIdentifier {
                context: "ExprAst::Var",
            });
        }
        Ok(Self::Var(name))
    }

    /// Suma (`lhs + rhs`).
    #[allow(clippy::should_implement_trait)]
    pub fn add(lhs: Self, rhs: Self) -> Self {
        Self::Add(Box::new(lhs), Box::new(rhs))
    }

    /// Resta (`lhs - rhs`).
    #[allow(clippy::should_implement_trait)]
    pub fn sub(lhs: Self, rhs: Self) -> Self {
        Self::Sub(Box::new(lhs), Box::new(rhs))
    }

    /// Multiplicación (`lhs * rhs`).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(lhs: Self, rhs: Self) -> Self {
        Self::Mul(Box::new(lhs), Box::new(rhs))
    }

    /// División (`lhs / rhs`).
    #[allow(clippy::should_implement_trait)]
    pub fn div(lhs: Self, rhs: Self) -> Self {
        Self::Div(Box::new(lhs), Box::new(rhs))
    }

    /// Menor estricto (`lhs < rhs`).
    pub fn lt(lhs: Self, rhs: Self) -> Self {
        Self::Lt(Box::new(lhs), Box::new(rhs))
    }

    /// Menor o igual (`lhs <= rhs`).
    pub fn le(lhs: Self, rhs: Self) -> Self {
        Self::Le(Box::new(lhs), Box::new(rhs))
    }

    /// Mayor estricto (`lhs > rhs`).
    pub fn gt(lhs: Self, rhs: Self) -> Self {
        Self::Gt(Box::new(lhs), Box::new(rhs))
    }

    /// Mayor o igual (`lhs >= rhs`).
    pub fn ge(lhs: Self, rhs: Self) -> Self {
        Self::Ge(Box::new(lhs), Box::new(rhs))
    }

    /// Igualdad (`lhs == rhs`).
    pub fn eq(lhs: Self, rhs: Self) -> Self {
        Self::Eq(Box::new(lhs), Box::new(rhs))
    }

    /// Desigualdad (`lhs != rhs`).
    pub fn ne(lhs: Self, rhs: Self) -> Self {
        Self::Ne(Box::new(lhs), Box::new(rhs))
    }

    /// Conjunción lógica (`lhs && rhs`).
    pub fn and(lhs: Self, rhs: Self) -> Self {
        Self::And(Box::new(lhs), Box::new(rhs))
    }

    /// Disyunción lógica (`lhs || rhs`).
    pub fn or(lhs: Self, rhs: Self) -> Self {
        Self::Or(Box::new(lhs), Box::new(rhs))
    }

    /// Negación lógica (`!expr`).
    #[allow(clippy::should_implement_trait)]
    pub fn not(expr: Self) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Valida de forma iterativa profundidad, nombres de variables y finitud de constantes.
    pub fn validate(&self) -> Result<(), SpecError> {
        let mut stack = vec![(self, 1usize)];
        while let Some((node, depth)) = stack.pop() {
            if depth > EXPR_AST_MAX_DEPTH {
                return Err(SpecError::InvalidFieldValue {
                    context: "ExprAst",
                    field: "depth",
                    reason: format!(
                        "expression depth {depth} exceeds maximum {EXPR_AST_MAX_DEPTH}"
                    ),
                });
            }
            match node {
                Self::ConstF64 { value, .. } => {
                    if !value.is_finite() {
                        return Err(SpecError::NonFiniteFloat {
                            context: "ExprAst::ConstF64".to_owned(),
                        });
                    }
                }
                Self::ConstBool(_) => {}
                Self::Var(name) => {
                    if name.trim().is_empty() {
                        return Err(SpecError::EmptyIdentifier {
                            context: "ExprAst::Var",
                        });
                    }
                }
                Self::Not(inner) => {
                    stack.push((inner, depth + 1));
                }
                Self::Add(lhs, rhs)
                | Self::Sub(lhs, rhs)
                | Self::Mul(lhs, rhs)
                | Self::Div(lhs, rhs)
                | Self::Lt(lhs, rhs)
                | Self::Le(lhs, rhs)
                | Self::Gt(lhs, rhs)
                | Self::Ge(lhs, rhs)
                | Self::Eq(lhs, rhs)
                | Self::Ne(lhs, rhs)
                | Self::And(lhs, rhs)
                | Self::Or(lhs, rhs) => {
                    stack.push((rhs, depth + 1));
                    stack.push((lhs, depth + 1));
                }
            }
        }
        Ok(())
    }

    /// Serializa el AST a JSON canónico tras validarlo.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_expr_json(&mut out, self, 1)?;
        Ok(out)
    }

    /// Deserializa un [`ExprAst`] desde JSON rechazando campos desconocidos.
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa un [`ExprAst`] desde JSON aplicando la política indicada para campos extra.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let ast = parse_expr_json(&value, policy, 1)?;
        ast.validate()?;
        Ok(ast)
    }
}

pub(crate) fn write_expr_json(
    out: &mut String,
    node: &ExprAst,
    depth: usize,
) -> Result<(), SpecError> {
    if depth > EXPR_AST_MAX_DEPTH {
        return Err(SpecError::InvalidFieldValue {
            context: "ExprAst",
            field: "depth",
            reason: format!("expression depth {depth} exceeds maximum {EXPR_AST_MAX_DEPTH}"),
        });
    }
    match node {
        ExprAst::ConstF64 { value, unit } => {
            out.push_str("{\"op\":\"const_f64\",\"unit\":");
            write_json_string(out, unit.as_str());
            out.push_str(",\"value\":");
            write_canonical_f64(out, *value)?;
            out.push('}');
        }
        ExprAst::ConstBool(b) => {
            out.push_str("{\"op\":\"const_bool\",\"value\":");
            out.push_str(if *b { "true" } else { "false" });
            out.push('}');
        }
        ExprAst::Var(name) => {
            out.push_str("{\"op\":\"var\",\"name\":");
            write_json_string(out, name);
            out.push('}');
        }
        ExprAst::Not(inner) => {
            out.push_str("{\"op\":\"not\",\"expr\":");
            write_expr_json(out, inner, depth + 1)?;
            out.push('}');
        }
        ExprAst::Add(lhs, rhs) => write_binary_json(out, "add", lhs, rhs, depth)?,
        ExprAst::Sub(lhs, rhs) => write_binary_json(out, "sub", lhs, rhs, depth)?,
        ExprAst::Mul(lhs, rhs) => write_binary_json(out, "mul", lhs, rhs, depth)?,
        ExprAst::Div(lhs, rhs) => write_binary_json(out, "div", lhs, rhs, depth)?,
        ExprAst::Lt(lhs, rhs) => write_binary_json(out, "lt", lhs, rhs, depth)?,
        ExprAst::Le(lhs, rhs) => write_binary_json(out, "le", lhs, rhs, depth)?,
        ExprAst::Gt(lhs, rhs) => write_binary_json(out, "gt", lhs, rhs, depth)?,
        ExprAst::Ge(lhs, rhs) => write_binary_json(out, "ge", lhs, rhs, depth)?,
        ExprAst::Eq(lhs, rhs) => write_binary_json(out, "eq", lhs, rhs, depth)?,
        ExprAst::Ne(lhs, rhs) => write_binary_json(out, "ne", lhs, rhs, depth)?,
        ExprAst::And(lhs, rhs) => write_binary_json(out, "and", lhs, rhs, depth)?,
        ExprAst::Or(lhs, rhs) => write_binary_json(out, "or", lhs, rhs, depth)?,
    }
    Ok(())
}

fn write_binary_json(
    out: &mut String,
    op: &str,
    lhs: &ExprAst,
    rhs: &ExprAst,
    depth: usize,
) -> Result<(), SpecError> {
    out.push_str("{\"op\":");
    write_json_string(out, op);
    out.push_str(",\"lhs\":");
    write_expr_json(out, lhs, depth + 1)?;
    out.push_str(",\"rhs\":");
    write_expr_json(out, rhs, depth + 1)?;
    out.push('}');
    Ok(())
}

pub(crate) fn parse_expr_json(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    depth: usize,
) -> Result<ExprAst, SpecError> {
    if depth > EXPR_AST_MAX_DEPTH {
        return Err(SpecError::InvalidFieldValue {
            context: "ExprAst",
            field: "depth",
            reason: format!("expression depth {depth} exceeds maximum {EXPR_AST_MAX_DEPTH}"),
        });
    }
    let obj = match value {
        JsonValue::Object(map) => map,
        _ => {
            return Err(SpecError::InvalidFieldValue {
                context: "ExprAst",
                field: "root",
                reason: "expected JSON object".to_owned(),
            });
        }
    };

    let op = match obj.get("op") {
        Some(JsonValue::String(s)) => s.as_str(),
        Some(_) => {
            return Err(SpecError::InvalidFieldValue {
                context: "ExprAst",
                field: "op",
                reason: "expected string".to_owned(),
            });
        }
        None => {
            return Err(SpecError::MissingField {
                context: "ExprAst",
                field: "op",
            });
        }
    };

    match op {
        "const_f64" => {
            reject_unknown_fields(obj, &["op", "value", "unit"], "ExprAst::ConstF64", policy)?;
            let number = match obj.get("value") {
                Some(JsonValue::Number(num)) => match num {
                    JsonNumber::I64(v) => *v as f64,
                    JsonNumber::U64(v) => *v as f64,
                    JsonNumber::F64(v) => *v,
                },
                Some(_) => {
                    return Err(SpecError::InvalidFieldValue {
                        context: "ExprAst::ConstF64",
                        field: "value",
                        reason: "expected finite number".to_owned(),
                    });
                }
                None => {
                    return Err(SpecError::MissingField {
                        context: "ExprAst::ConstF64",
                        field: "value",
                    });
                }
            };
            let unit = match obj.get("unit") {
                Some(JsonValue::String(u)) => ExprUnit::parse(u)?,
                Some(_) => {
                    return Err(SpecError::InvalidFieldValue {
                        context: "ExprAst::ConstF64",
                        field: "unit",
                        reason: "expected string".to_owned(),
                    });
                }
                None => ExprUnit::Dimensionless,
            };
            ExprAst::const_f64_with_unit(number, unit)
        }
        "const_bool" => {
            reject_unknown_fields(obj, &["op", "value"], "ExprAst::ConstBool", policy)?;
            let b = match obj.get("value") {
                Some(JsonValue::Bool(b)) => *b,
                Some(_) => {
                    return Err(SpecError::InvalidFieldValue {
                        context: "ExprAst::ConstBool",
                        field: "value",
                        reason: "expected boolean".to_owned(),
                    });
                }
                None => {
                    return Err(SpecError::MissingField {
                        context: "ExprAst::ConstBool",
                        field: "value",
                    });
                }
            };
            Ok(ExprAst::ConstBool(b))
        }
        "var" => {
            reject_unknown_fields(obj, &["op", "name"], "ExprAst::Var", policy)?;
            let name = match obj.get("name") {
                Some(JsonValue::String(s)) => s.clone(),
                Some(_) => {
                    return Err(SpecError::InvalidFieldValue {
                        context: "ExprAst::Var",
                        field: "name",
                        reason: "expected string".to_owned(),
                    });
                }
                None => {
                    return Err(SpecError::MissingField {
                        context: "ExprAst::Var",
                        field: "name",
                    });
                }
            };
            ExprAst::var(name)
        }
        "not" => {
            reject_unknown_fields(obj, &["op", "expr"], "ExprAst::Not", policy)?;
            let inner = obj.get("expr").ok_or(SpecError::MissingField {
                context: "ExprAst::Not",
                field: "expr",
            })?;
            Ok(ExprAst::not(parse_expr_json(inner, policy, depth + 1)?))
        }
        "add" | "sub" | "mul" | "div" | "lt" | "le" | "gt" | "ge" | "eq" | "ne" | "and" | "or" => {
            reject_unknown_fields(obj, &["op", "lhs", "rhs"], "ExprAst::Binary", policy)?;
            let lhs_val = obj.get("lhs").ok_or(SpecError::MissingField {
                context: "ExprAst::Binary",
                field: "lhs",
            })?;
            let rhs_val = obj.get("rhs").ok_or(SpecError::MissingField {
                context: "ExprAst::Binary",
                field: "rhs",
            })?;
            let lhs = parse_expr_json(lhs_val, policy, depth + 1)?;
            let rhs = parse_expr_json(rhs_val, policy, depth + 1)?;
            Ok(match op {
                "add" => ExprAst::add(lhs, rhs),
                "sub" => ExprAst::sub(lhs, rhs),
                "mul" => ExprAst::mul(lhs, rhs),
                "div" => ExprAst::div(lhs, rhs),
                "lt" => ExprAst::lt(lhs, rhs),
                "le" => ExprAst::le(lhs, rhs),
                "gt" => ExprAst::gt(lhs, rhs),
                "ge" => ExprAst::ge(lhs, rhs),
                "eq" => ExprAst::eq(lhs, rhs),
                "ne" => ExprAst::ne(lhs, rhs),
                "and" => ExprAst::and(lhs, rhs),
                "or" => ExprAst::or(lhs, rhs),
                _ => unreachable!("matched above"),
            })
        }
        "eval" | "exec" | "script" | "code" | "lambda" | "call" | "js" | "python" | "kotlin"
        | "rust" => Err(SpecError::CodeEvalRejected {
            context: "ExprAst",
            field: "op".to_owned(),
            reason: format!("arbitrary code evaluation operator '{op}' is forbidden"),
        }),
        other => Err(SpecError::InvalidFieldValue {
            context: "ExprAst",
            field: "op",
            reason: format!("unknown expression operator '{other}'"),
        }),
    }
}

fn reject_unknown_fields(
    obj: &BTreeMap<String, JsonValue>,
    allowed: &[&str],
    context: &'static str,
    policy: UnknownFieldPolicy,
) -> Result<(), SpecError> {
    if policy == UnknownFieldPolicy::Reject {
        for key in obj.keys() {
            if !allowed.contains(&key.as_str()) {
                return Err(SpecError::UnknownField {
                    context,
                    field: key.clone(),
                });
            }
        }
    }
    Ok(())
}
