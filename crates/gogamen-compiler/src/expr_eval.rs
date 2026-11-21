//! Compilación por índice y evaluación acotada de expresiones simbólicas [`ExprAst`].
//!
//! Este módulo resuelve todas las variables a [`VarIndex`] en tiempo de compilación,
//! valida tipos y compatibilidad de unidades físicas, rechaza constantes no finitas
//! y divisiones por cero, y ejecuta un bytecode plano en [`BoundedExprEvaluator`] sin
//! búsquedas por string, sin recursión ilimitada, sin IO y sin callbacks externos.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_spec::{EXPR_AST_MAX_DEPTH, ExprAst, ExprType, ExprUnit};

/// Índice numérico pre-resuelto de una variable en [`ExprSchema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarIndex(pub usize);

impl VarIndex {
    /// Crea un índice de variable.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Devuelve el índice subyacente.
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for VarIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "var#{}", self.0)
    }
}

/// Valor en tiempo de evaluación para un slot de variable o resultado de expresión.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExprValue {
    /// Escalar de punto flotante (`f64`).
    Scalar(f64),
    /// Valor booleano.
    Bool(bool),
}

impl ExprValue {
    /// Devuelve el valor `f64` si es [`ExprValue::Scalar`].
    pub const fn as_f64(self) -> Option<f64> {
        match self {
            Self::Scalar(v) => Some(v),
            Self::Bool(_) => None,
        }
    }

    /// Devuelve el valor `bool` si es [`ExprValue::Bool`].
    pub const fn as_bool(self) -> Option<bool> {
        match self {
            Self::Scalar(_) => None,
            Self::Bool(b) => Some(b),
        }
    }

    const fn kind_name(self) -> &'static str {
        match self {
            Self::Scalar(_) => "Scalar",
            Self::Bool(_) => "Bool",
        }
    }
}

/// Registro de slots tipados que mapea nombres de variables a `(VarIndex, ExprType)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VariableSlotRegistry {
    by_name: BTreeMap<String, (VarIndex, ExprType)>,
    slot_types: Vec<ExprType>,
}

/// Alias de esquema de variables para compilación de expresiones simbólicas.
pub type ExprSchema = VariableSlotRegistry;

impl VariableSlotRegistry {
    /// Crea un registro vacío de variables.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una variable con su [`ExprType`] y devuelve su [`VarIndex`] asignado.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        ty: ExprType,
    ) -> Result<VarIndex, ExprCompileError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ExprCompileError::EmptyVariableName);
        }
        if self.by_name.contains_key(&name) {
            return Err(ExprCompileError::DuplicateVariable { name });
        }
        let idx = VarIndex::new(self.slot_types.len());
        self.slot_types.push(ty);
        self.by_name.insert(name, (idx, ty));
        Ok(idx)
    }

    /// Resuelve una variable por nombre a `(VarIndex, ExprType)`.
    pub fn resolve(&self, name: &str) -> Option<(VarIndex, ExprType)> {
        self.by_name.get(name).copied()
    }

    /// Número total de slots registrados.
    pub fn len(&self) -> usize {
        self.slot_types.len()
    }

    /// Indica si el registro está vacío.
    pub fn is_empty(&self) -> bool {
        self.slot_types.is_empty()
    }

    /// Devuelve los tipos registrados indexados por [`VarIndex`].
    pub fn slot_types(&self) -> &[ExprType] {
        &self.slot_types
    }
}

/// Errores de compilación y tipado estático de [`ExprAst`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprCompileError {
    /// Nombre de variable vacío o compuesto solo por espacios.
    EmptyVariableName,
    /// Variable duplicada al registrar en [`VariableSlotRegistry`].
    DuplicateVariable {
        /// Nombre duplicado.
        name: String,
    },
    /// Variable referenciada en el AST que no existe en el [`ExprSchema`].
    UnknownVariable {
        /// Nombre de la variable ausente.
        name: String,
    },
    /// Constante numérica no finita (`NaN` o infinito).
    NonFiniteConstant,
    /// Profundidad del AST excede el límite permitido.
    DepthLimitExceeded {
        /// Profundidad observada.
        depth: usize,
        /// Profundidad máxima permitida.
        max_depth: usize,
    },
    /// Tipo incompatible para el operador.
    TypeMismatch {
        /// Operador donde ocurrió el error.
        op: &'static str,
        /// Tipo esperado.
        expected: &'static str,
        /// Tipo encontrado.
        found: ExprType,
    },
    /// Unidades físicas incompatibles para el operador.
    IncompatibleUnits {
        /// Operador donde ocurrió el error.
        op: &'static str,
        /// Unidad izquierda.
        lhs: ExprUnit,
        /// Unidad derecha.
        rhs: ExprUnit,
    },
    /// División por constante cero detectada en compilación.
    DivisionByZero,
}

impl fmt::Display for ExprCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVariableName => f.write_str("variable name must not be empty"),
            Self::DuplicateVariable { name } => {
                write!(f, "duplicate variable '{name}' in ExprSchema")
            }
            Self::UnknownVariable { name } => {
                write!(f, "unknown variable '{name}' in expression")
            }
            Self::NonFiniteConstant => f.write_str("non-finite constant in expression"),
            Self::DepthLimitExceeded { depth, max_depth } => {
                write!(f, "expression depth {depth} exceeds maximum {max_depth}")
            }
            Self::TypeMismatch {
                op,
                expected,
                found,
            } => write!(
                f,
                "type mismatch in '{op}': expected {expected}, found {found}"
            ),
            Self::IncompatibleUnits { op, lhs, rhs } => {
                write!(f, "incompatible units in '{op}': {lhs} and {rhs}")
            }
            Self::DivisionByZero => f.write_str("division by zero constant in expression"),
        }
    }
}

impl Error for ExprCompileError {}

/// Errores en tiempo de evaluación acotada por [`BoundedExprEvaluator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprEvalError {
    /// Se superó el presupuesto máximo de pasos (`max_steps`).
    StepLimitExceeded {
        /// Límite configurado.
        max_steps: usize,
    },
    /// Se superó la profundidad máxima de pila de evaluación.
    StackDepthExceeded {
        /// Profundidad máxima permitida.
        max_depth: usize,
    },
    /// El slice `values` no contiene el slot requerido.
    MissingVariableSlot {
        /// Índice requerido.
        index: VarIndex,
        /// Longitud del slice entregado.
        values_len: usize,
    },
    /// El valor en `values[index]` no coincide con el [`ExprType`] compilado.
    VariableTypeMismatch {
        /// Índice afectado.
        index: VarIndex,
        /// Tipo esperado.
        expected: ExprType,
        /// Variante encontrada.
        found: &'static str,
    },
    /// Valor no finito (`NaN` o infinito) en entrada o resultado intermedio.
    NonFiniteValue {
        /// Contexto donde apareció el valor no finito.
        context: &'static str,
    },
    /// División por cero en tiempo de evaluación.
    DivisionByZero,
    /// Invariante interno de bytecode inválido.
    InvalidBytecode,
}

impl fmt::Display for ExprEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepLimitExceeded { max_steps } => {
                write!(f, "expression evaluation exceeded max_steps ({max_steps})")
            }
            Self::StackDepthExceeded { max_depth } => {
                write!(f, "expression stack depth exceeded limit ({max_depth})")
            }
            Self::MissingVariableSlot { index, values_len } => write!(
                f,
                "missing variable slot {index} (values length is {values_len})"
            ),
            Self::VariableTypeMismatch {
                index,
                expected,
                found,
            } => write!(
                f,
                "variable type mismatch at {index}: expected {expected}, found {found}"
            ),
            Self::NonFiniteValue { context } => {
                write!(f, "non-finite f64 value during {context}")
            }
            Self::DivisionByZero => f.write_str("division by zero during expression evaluation"),
            Self::InvalidBytecode => f.write_str("invalid compiled expression bytecode"),
        }
    }
}

impl Error for ExprEvalError {}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExprOp {
    PushF64(f64),
    PushBool(bool),
    LoadVar(VarIndex),
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinaryOpKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::And => "and",
            Self::Or => "or",
        }
    }

    const fn to_op(self) -> ExprOp {
        match self {
            Self::Add => ExprOp::Add,
            Self::Sub => ExprOp::Sub,
            Self::Mul => ExprOp::Mul,
            Self::Div => ExprOp::Div,
            Self::Lt => ExprOp::Lt,
            Self::Le => ExprOp::Le,
            Self::Gt => ExprOp::Gt,
            Self::Ge => ExprOp::Ge,
            Self::Eq => ExprOp::Eq,
            Self::Ne => ExprOp::Ne,
            Self::And => ExprOp::And,
            Self::Or => ExprOp::Or,
        }
    }
}

enum CompileFrame<'a> {
    Visit(&'a ExprAst, usize),
    EmitNot,
    EmitBinary {
        kind: BinaryOpKind,
        rhs_is_const_zero: bool,
    },
}

/// Expresión simbólica pre-compilada, tipada y con variables resueltas a [`VarIndex`].
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledExpr {
    ops: Vec<ExprOp>,
    result_type: ExprType,
    max_stack_depth: usize,
    required_slots: Vec<(VarIndex, ExprType)>,
}

impl CompiledExpr {
    /// Compila un [`ExprAst`] contra un [`ExprSchema`] usando [`EXPR_AST_MAX_DEPTH`].
    pub fn compile(ast: &ExprAst, schema: &ExprSchema) -> Result<Self, ExprCompileError> {
        Self::compile_with_max_depth(ast, schema, EXPR_AST_MAX_DEPTH)
    }

    /// Compila un [`ExprAst`] contra un [`ExprSchema`] con un límite explícito de profundidad.
    pub fn compile_with_max_depth(
        ast: &ExprAst,
        schema: &ExprSchema,
        max_depth: usize,
    ) -> Result<Self, ExprCompileError> {
        let mut frames = vec![CompileFrame::Visit(ast, 1)];
        let mut ops = Vec::new();
        let mut type_stack = Vec::new();
        let mut required_map = BTreeMap::new();
        let mut stack_depth = 0usize;
        let mut max_stack_depth = 0usize;

        while let Some(frame) = frames.pop() {
            match frame {
                CompileFrame::Visit(node, depth) => {
                    if depth > max_depth {
                        return Err(ExprCompileError::DepthLimitExceeded { depth, max_depth });
                    }
                    match node {
                        ExprAst::ConstF64 { value, unit } => {
                            if !value.is_finite() {
                                return Err(ExprCompileError::NonFiniteConstant);
                            }
                            let normalized = if *value == 0.0 { 0.0 } else { *value };
                            ops.push(ExprOp::PushF64(normalized));
                            type_stack.push(ExprType::Scalar(*unit));
                            stack_depth += 1;
                            max_stack_depth = max_stack_depth.max(stack_depth);
                        }
                        ExprAst::ConstBool(b) => {
                            ops.push(ExprOp::PushBool(*b));
                            type_stack.push(ExprType::Bool);
                            stack_depth += 1;
                            max_stack_depth = max_stack_depth.max(stack_depth);
                        }
                        ExprAst::Var(name) => {
                            if name.trim().is_empty() {
                                return Err(ExprCompileError::EmptyVariableName);
                            }
                            let (idx, ty) = schema.resolve(name).ok_or_else(|| {
                                ExprCompileError::UnknownVariable { name: name.clone() }
                            })?;
                            ops.push(ExprOp::LoadVar(idx));
                            type_stack.push(ty);
                            required_map.entry(idx).or_insert(ty);
                            stack_depth += 1;
                            max_stack_depth = max_stack_depth.max(stack_depth);
                        }
                        ExprAst::Not(inner) => {
                            frames.push(CompileFrame::EmitNot);
                            frames.push(CompileFrame::Visit(inner, depth + 1));
                        }
                        ExprAst::Add(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Add, lhs, rhs, depth);
                        }
                        ExprAst::Sub(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Sub, lhs, rhs, depth);
                        }
                        ExprAst::Mul(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Mul, lhs, rhs, depth);
                        }
                        ExprAst::Div(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Div, lhs, rhs, depth);
                        }
                        ExprAst::Lt(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Lt, lhs, rhs, depth);
                        }
                        ExprAst::Le(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Le, lhs, rhs, depth);
                        }
                        ExprAst::Gt(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Gt, lhs, rhs, depth);
                        }
                        ExprAst::Ge(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Ge, lhs, rhs, depth);
                        }
                        ExprAst::Eq(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Eq, lhs, rhs, depth);
                        }
                        ExprAst::Ne(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Ne, lhs, rhs, depth);
                        }
                        ExprAst::And(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::And, lhs, rhs, depth);
                        }
                        ExprAst::Or(lhs, rhs) => {
                            push_binary(&mut frames, BinaryOpKind::Or, lhs, rhs, depth);
                        }
                    }
                }
                CompileFrame::EmitNot => {
                    let inner_ty = type_stack
                        .pop()
                        .ok_or(ExprCompileError::NonFiniteConstant)?;
                    if inner_ty != ExprType::Bool {
                        return Err(ExprCompileError::TypeMismatch {
                            op: "not",
                            expected: "Bool",
                            found: inner_ty,
                        });
                    }
                    ops.push(ExprOp::Not);
                    type_stack.push(ExprType::Bool);
                }
                CompileFrame::EmitBinary {
                    kind,
                    rhs_is_const_zero,
                } => {
                    let rhs_ty = type_stack
                        .pop()
                        .ok_or(ExprCompileError::NonFiniteConstant)?;
                    let lhs_ty = type_stack
                        .pop()
                        .ok_or(ExprCompileError::NonFiniteConstant)?;
                    if kind == BinaryOpKind::Div && rhs_is_const_zero {
                        return Err(ExprCompileError::DivisionByZero);
                    }
                    let out_ty = check_binary_types(kind, lhs_ty, rhs_ty)?;
                    ops.push(kind.to_op());
                    type_stack.push(out_ty);
                    stack_depth -= 1;
                }
            }
        }

        let result_type = type_stack
            .pop()
            .ok_or(ExprCompileError::NonFiniteConstant)?;
        let required_slots = required_map.into_iter().collect();

        Ok(Self {
            ops,
            result_type,
            max_stack_depth,
            required_slots,
        })
    }

    /// Tipo estático resultante de evaluar la expresión compilada.
    pub const fn result_type(&self) -> ExprType {
        self.result_type
    }

    /// Número de instrucciones en el bytecode compilado.
    pub fn instruction_count(&self) -> usize {
        self.ops.len()
    }

    /// Profundidad máxima de pila requerida durante la evaluación.
    pub const fn max_stack_depth(&self) -> usize {
        self.max_stack_depth
    }

    /// Slots de variables requeridos por esta expresión junto con su tipo esperado.
    pub fn required_slots(&self) -> &[(VarIndex, ExprType)] {
        &self.required_slots
    }
}

fn push_binary<'a>(
    frames: &mut Vec<CompileFrame<'a>>,
    kind: BinaryOpKind,
    lhs: &'a ExprAst,
    rhs: &'a ExprAst,
    depth: usize,
) {
    let rhs_is_const_zero = matches!(rhs, ExprAst::ConstF64 { value, .. } if *value == 0.0);
    frames.push(CompileFrame::EmitBinary {
        kind,
        rhs_is_const_zero,
    });
    frames.push(CompileFrame::Visit(rhs, depth + 1));
    frames.push(CompileFrame::Visit(lhs, depth + 1));
}

fn check_binary_types(
    kind: BinaryOpKind,
    lhs: ExprType,
    rhs: ExprType,
) -> Result<ExprType, ExprCompileError> {
    let op = kind.name();
    match kind {
        BinaryOpKind::Add | BinaryOpKind::Sub => {
            let lhs_u = expect_scalar(op, lhs)?;
            let rhs_u = expect_scalar(op, rhs)?;
            if lhs_u != rhs_u {
                return Err(ExprCompileError::IncompatibleUnits {
                    op,
                    lhs: lhs_u,
                    rhs: rhs_u,
                });
            }
            Ok(ExprType::Scalar(lhs_u))
        }
        BinaryOpKind::Mul => {
            let lhs_u = expect_scalar(op, lhs)?;
            let rhs_u = expect_scalar(op, rhs)?;
            let out_u = lhs_u
                .combine_mul(rhs_u)
                .ok_or(ExprCompileError::IncompatibleUnits {
                    op,
                    lhs: lhs_u,
                    rhs: rhs_u,
                })?;
            Ok(ExprType::Scalar(out_u))
        }
        BinaryOpKind::Div => {
            let lhs_u = expect_scalar(op, lhs)?;
            let rhs_u = expect_scalar(op, rhs)?;
            let out_u = lhs_u
                .combine_div(rhs_u)
                .ok_or(ExprCompileError::IncompatibleUnits {
                    op,
                    lhs: lhs_u,
                    rhs: rhs_u,
                })?;
            Ok(ExprType::Scalar(out_u))
        }
        BinaryOpKind::Lt | BinaryOpKind::Le | BinaryOpKind::Gt | BinaryOpKind::Ge => {
            let lhs_u = expect_scalar(op, lhs)?;
            let rhs_u = expect_scalar(op, rhs)?;
            if lhs_u != rhs_u {
                return Err(ExprCompileError::IncompatibleUnits {
                    op,
                    lhs: lhs_u,
                    rhs: rhs_u,
                });
            }
            Ok(ExprType::Bool)
        }
        BinaryOpKind::Eq | BinaryOpKind::Ne => match (lhs, rhs) {
            (ExprType::Scalar(lhs_u), ExprType::Scalar(rhs_u)) => {
                if lhs_u != rhs_u {
                    return Err(ExprCompileError::IncompatibleUnits {
                        op,
                        lhs: lhs_u,
                        rhs: rhs_u,
                    });
                }
                Ok(ExprType::Bool)
            }
            (ExprType::Bool, ExprType::Bool) => Ok(ExprType::Bool),
            (ExprType::Bool, other) => Err(ExprCompileError::TypeMismatch {
                op,
                expected: "Bool",
                found: other,
            }),
            (ExprType::Scalar(_), other) => Err(ExprCompileError::TypeMismatch {
                op,
                expected: "Scalar",
                found: other,
            }),
        },
        BinaryOpKind::And | BinaryOpKind::Or => {
            if lhs != ExprType::Bool {
                return Err(ExprCompileError::TypeMismatch {
                    op,
                    expected: "Bool",
                    found: lhs,
                });
            }
            if rhs != ExprType::Bool {
                return Err(ExprCompileError::TypeMismatch {
                    op,
                    expected: "Bool",
                    found: rhs,
                });
            }
            Ok(ExprType::Bool)
        }
    }
}

fn expect_scalar(op: &'static str, ty: ExprType) -> Result<ExprUnit, ExprCompileError> {
    match ty {
        ExprType::Scalar(u) => Ok(u),
        ExprType::Bool => Err(ExprCompileError::TypeMismatch {
            op,
            expected: "Scalar",
            found: ty,
        }),
    }
}

/// Evaluador iterativo acotado por `max_steps` y profundidad máxima de pila.
///
/// No realiza búsquedas por nombre de variable, no usa recursión, no ejecuta IO
/// ni invoca callbacks externos.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundedExprEvaluator;

impl BoundedExprEvaluator {
    /// Evalúa una expresión pre-compilada [`CompiledExpr`] usando el slice de valores `values`
    /// indexado por [`VarIndex`] y respetando el presupuesto `max_steps`.
    ///
    /// No realiza asignaciones de heap: la pila de evaluación es un array fijo de
    /// `EXPR_AST_MAX_DEPTH + 1` valores en la pila del llamador. El límite lógico sigue
    /// siendo [`CompiledExpr::max_stack_depth`]; si un programa compilado con
    /// `compile_with_max_depth` excediera la capacidad fija, se devuelve
    /// [`ExprEvalError::StackDepthExceeded`].
    pub fn eval(
        compiled: &CompiledExpr,
        values: &[ExprValue],
        max_steps: usize,
    ) -> Result<ExprValue, ExprEvalError> {
        for &(slot, expected_ty) in &compiled.required_slots {
            let val = values
                .get(slot.as_usize())
                .ok_or(ExprEvalError::MissingVariableSlot {
                    index: slot,
                    values_len: values.len(),
                })?;
            match (expected_ty, *val) {
                (ExprType::Scalar(_), ExprValue::Scalar(v)) => {
                    if !v.is_finite() {
                        return Err(ExprEvalError::NonFiniteValue {
                            context: "variable input",
                        });
                    }
                }
                (ExprType::Bool, ExprValue::Bool(_)) => {}
                (expected, other) => {
                    return Err(ExprEvalError::VariableTypeMismatch {
                        index: slot,
                        expected,
                        found: other.kind_name(),
                    });
                }
            }
        }

        let max_stack = compiled.max_stack_depth;
        let mut stack = ExprStack::new();
        let mut steps = 0usize;

        for &op in &compiled.ops {
            steps += 1;
            if steps > max_steps {
                return Err(ExprEvalError::StepLimitExceeded { max_steps });
            }

            match op {
                ExprOp::PushF64(v) => {
                    push_checked(&mut stack, ExprValue::Scalar(v), max_stack)?;
                }
                ExprOp::PushBool(b) => {
                    push_checked(&mut stack, ExprValue::Bool(b), max_stack)?;
                }
                ExprOp::LoadVar(idx) => {
                    let v = values.get(idx.as_usize()).copied().ok_or(
                        ExprEvalError::MissingVariableSlot {
                            index: idx,
                            values_len: values.len(),
                        },
                    )?;
                    push_checked(&mut stack, v, max_stack)?;
                }
                ExprOp::Not => {
                    let val = pop_bool(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(!val), max_stack)?;
                }
                ExprOp::Add => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    let res = ensure_finite(lhs + rhs, "addition")?;
                    push_checked(&mut stack, ExprValue::Scalar(res), max_stack)?;
                }
                ExprOp::Sub => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    let res = ensure_finite(lhs - rhs, "subtraction")?;
                    push_checked(&mut stack, ExprValue::Scalar(res), max_stack)?;
                }
                ExprOp::Mul => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    let res = ensure_finite(lhs * rhs, "multiplication")?;
                    push_checked(&mut stack, ExprValue::Scalar(res), max_stack)?;
                }
                ExprOp::Div => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    if rhs == 0.0 {
                        return Err(ExprEvalError::DivisionByZero);
                    }
                    let res = ensure_finite(lhs / rhs, "division")?;
                    push_checked(&mut stack, ExprValue::Scalar(res), max_stack)?;
                }
                ExprOp::Lt => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs < rhs), max_stack)?;
                }
                ExprOp::Le => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs <= rhs), max_stack)?;
                }
                ExprOp::Gt => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs > rhs), max_stack)?;
                }
                ExprOp::Ge => {
                    let (lhs, rhs) = pop_two_f64(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs >= rhs), max_stack)?;
                }
                ExprOp::Eq => {
                    let rhs = stack.pop().ok_or(ExprEvalError::InvalidBytecode)?;
                    let lhs = stack.pop().ok_or(ExprEvalError::InvalidBytecode)?;
                    let eq = match (lhs, rhs) {
                        (ExprValue::Scalar(a), ExprValue::Scalar(b)) => a == b,
                        (ExprValue::Bool(a), ExprValue::Bool(b)) => a == b,
                        _ => return Err(ExprEvalError::InvalidBytecode),
                    };
                    push_checked(&mut stack, ExprValue::Bool(eq), max_stack)?;
                }
                ExprOp::Ne => {
                    let rhs = stack.pop().ok_or(ExprEvalError::InvalidBytecode)?;
                    let lhs = stack.pop().ok_or(ExprEvalError::InvalidBytecode)?;
                    let ne = match (lhs, rhs) {
                        (ExprValue::Scalar(a), ExprValue::Scalar(b)) => a != b,
                        (ExprValue::Bool(a), ExprValue::Bool(b)) => a != b,
                        _ => return Err(ExprEvalError::InvalidBytecode),
                    };
                    push_checked(&mut stack, ExprValue::Bool(ne), max_stack)?;
                }
                ExprOp::And => {
                    let rhs = pop_bool(&mut stack)?;
                    let lhs = pop_bool(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs && rhs), max_stack)?;
                }
                ExprOp::Or => {
                    let rhs = pop_bool(&mut stack)?;
                    let lhs = pop_bool(&mut stack)?;
                    push_checked(&mut stack, ExprValue::Bool(lhs || rhs), max_stack)?;
                }
            }
        }

        if stack.len != 1 {
            return Err(ExprEvalError::InvalidBytecode);
        }
        stack.pop().ok_or(ExprEvalError::InvalidBytecode)
    }
}

fn push_checked(
    stack: &mut ExprStack,
    val: ExprValue,
    max_depth: usize,
) -> Result<(), ExprEvalError> {
    if stack.len >= max_depth || !stack.push(val) {
        return Err(ExprEvalError::StackDepthExceeded { max_depth });
    }
    Ok(())
}

/// Capacidad de la pila de evaluación: `EXPR_AST_MAX_DEPTH + 1` valores.
const EXPR_STACK_CAPACITY: usize = EXPR_AST_MAX_DEPTH + 1;

/// Pila de valores de capacidad fija, alojada en la pila del llamador (sin heap).
struct ExprStack {
    buf: [ExprValue; EXPR_STACK_CAPACITY],
    len: usize,
}

impl ExprStack {
    const fn new() -> Self {
        Self {
            buf: [ExprValue::Bool(false); EXPR_STACK_CAPACITY],
            len: 0,
        }
    }

    /// Apila `val`; devuelve `false` si la capacidad fija está agotada.
    fn push(&mut self, val: ExprValue) -> bool {
        match self.buf.get_mut(self.len) {
            Some(slot) => {
                *slot = val;
                self.len += 1;
                true
            }
            None => false,
        }
    }

    fn pop(&mut self) -> Option<ExprValue> {
        self.len = self.len.checked_sub(1)?;
        self.buf.get(self.len).copied()
    }
}

fn pop_bool(stack: &mut ExprStack) -> Result<bool, ExprEvalError> {
    match stack.pop() {
        Some(ExprValue::Bool(b)) => Ok(b),
        _ => Err(ExprEvalError::InvalidBytecode),
    }
}

fn pop_two_f64(stack: &mut ExprStack) -> Result<(f64, f64), ExprEvalError> {
    let rhs = match stack.pop() {
        Some(ExprValue::Scalar(v)) => v,
        _ => return Err(ExprEvalError::InvalidBytecode),
    };
    let lhs = match stack.pop() {
        Some(ExprValue::Scalar(v)) => v,
        _ => return Err(ExprEvalError::InvalidBytecode),
    };
    Ok((lhs, rhs))
}

fn ensure_finite(value: f64, context: &'static str) -> Result<f64, ExprEvalError> {
    if !value.is_finite() {
        return Err(ExprEvalError::NonFiniteValue { context });
    }
    Ok(if value == 0.0 { 0.0 } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> ExprSchema {
        let mut s = ExprSchema::new();
        s.register("x", ExprType::Scalar(ExprUnit::Ratio)).unwrap();
        s.register("y", ExprType::Scalar(ExprUnit::Ratio)).unwrap();
        s.register("b", ExprType::Bool).unwrap();
        s
    }

    fn vals() -> [ExprValue; 3] {
        [
            ExprValue::Scalar(3.0),
            ExprValue::Scalar(4.0),
            ExprValue::Bool(true),
        ]
    }

    fn run(ast: &ExprAst) -> Result<ExprValue, ExprEvalError> {
        let c = CompiledExpr::compile(ast, &schema()).unwrap();
        BoundedExprEvaluator::eval(&c, &vals(), 10_000)
    }

    fn var(n: &str) -> ExprAst {
        ExprAst::var(n).unwrap()
    }

    #[test]
    fn fixed_stack_eval_matches_expected_results() {
        let a = ExprAst::add(var("x"), ExprAst::mul(var("y"), var("x")));
        assert_eq!(run(&a), Ok(ExprValue::Scalar(15.0)));
        let b = ExprAst::and(
            ExprAst::lt(var("x"), var("y")),
            ExprAst::not(ExprAst::not(var("b"))),
        );
        assert_eq!(run(&b), Ok(ExprValue::Bool(true)));
        let c = ExprAst::eq(var("b"), ExprAst::not(var("b")));
        assert_eq!(run(&c), Ok(ExprValue::Bool(false)));
        let d = ExprAst::div(var("x"), ExprAst::sub(var("y"), var("y")));
        assert!(run(&d).is_err());
    }

    #[test]
    fn fixed_stack_eval_handles_max_depth_right_leaning_chain() {
        // Cadena derecha: x + (x + (x + ...)); pila máxima = profundidad del AST.
        let mut ast = var("x");
        for _ in 0..(EXPR_AST_MAX_DEPTH - 1) {
            ast = ExprAst::add(var("x"), ast);
        }
        let c = CompiledExpr::compile(&ast, &schema()).unwrap();
        assert!(c.max_stack_depth() <= EXPR_STACK_CAPACITY);
        let got = BoundedExprEvaluator::eval(&c, &vals(), 10_000).unwrap();
        assert_eq!(got, ExprValue::Scalar(3.0 * EXPR_AST_MAX_DEPTH as f64));
    }

    #[test]
    fn fixed_stack_overflow_reports_same_error_and_never_panics() {
        let mut st = ExprStack::new();
        for _ in 0..EXPR_STACK_CAPACITY {
            assert!(st.push(ExprValue::Bool(true)));
        }
        assert!(!st.push(ExprValue::Bool(true)));
        assert_eq!(
            push_checked(&mut st, ExprValue::Bool(true), EXPR_STACK_CAPACITY + 10),
            Err(ExprEvalError::StackDepthExceeded {
                max_depth: EXPR_STACK_CAPACITY + 10
            })
        );
        let mut small = ExprStack::new();
        assert!(push_checked(&mut small, ExprValue::Bool(true), 1).is_ok());
        assert_eq!(
            push_checked(&mut small, ExprValue::Bool(true), 1),
            Err(ExprEvalError::StackDepthExceeded { max_depth: 1 })
        );
        assert_eq!(small.pop(), Some(ExprValue::Bool(true)));
        assert_eq!(small.pop(), None);
    }
}
