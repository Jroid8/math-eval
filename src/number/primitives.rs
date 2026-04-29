use crate::{
    number::{Number, StdFloatConstsNameTrie},
    tokenizer::StandardFloatRecognizer,
};

macro_rules! impl_primitive {
    ($t: ident) => {
        impl Number for $t {
            type AsArg<'a> = Self;
            type Recognizer = StandardFloatRecognizer;
            type ConstsNameTrieType = StdFloatConstsNameTrie<Self>;

            const CONSTS_NAME_TRIE: StdFloatConstsNameTrie<Self> = StdFloatConstsNameTrie {
                pi: &std::$t::consts::PI,
                e: &std::$t::consts::E,
                tau: &std::$t::consts::TAU,
            };

            fn one() -> Self {
                1.0
            }

            fn zero() -> Self {
                0.0
            }

            fn asarg(&self) -> Self::AsArg<'_> {
                *self
            }

            fn is_one(value: Self::AsArg<'_>) -> bool {
                value == 1.0
            }

            fn is_two(value: Self::AsArg<'_>) -> bool {
                value == 2.0
            }

            fn is_ten(value: Self::AsArg<'_>) -> bool {
                value == 10.0
            }

            fn pow(self, rhs: Self) -> Self {
                self.powf(rhs)
            }

            fn modulo(self, rhs: Self) -> Self {
                self.rem_euclid(rhs)
            }

            fn sin(self) -> Self {
                self.sin()
            }

            fn cos(self) -> Self {
                self.cos()
            }

            fn tan(self) -> Self {
                self.tan()
            }

            fn cot(self) -> Self {
                // FIX: replace with a more accurate version
                let (sin, cos) = self.sin_cos();
                cos / sin
            }

            fn sinh(self) -> Self {
                self.sinh()
            }

            fn cosh(self) -> Self {
                self.cosh()
            }

            fn tanh(self) -> Self {
                self.tanh()
            }

            fn coth(self) -> Self {
                self.cosh() / self.sinh()
            }

            fn atan2(self, y: Self) -> Self {
                self.atan2(y)
            }

            fn asin(self) -> Self {
                self.asin()
            }

            fn acos(self) -> Self {
                self.acos()
            }

            fn atan(self) -> Self {
                self.atan()
            }

            fn acot(self) -> Self {
                (-self).atan() + std::$t::consts::FRAC_PI_2
            }

            fn asinh(self) -> Self {
                self.asinh()
            }

            fn acosh(self) -> Self {
                self.acosh()
            }

            fn atanh(self) -> Self {
                self.atanh()
            }

            fn acoth(self) -> Self {
                self.recip().atanh()
            }

            fn erf(self) -> Self {
                libm::Libm::<Self>::erf(self)
            }

            fn erfc(self) -> Self {
                libm::Libm::<Self>::erfc(self)
            }

            fn log(self, base: Self) -> Self {
                self.log(base)
            }

            fn log2(self) -> Self {
                self.log2()
            }

            fn log10(self) -> Self {
                self.log10()
            }

            fn ln(self) -> Self {
                self.ln()
            }

            fn ln_1p(self) -> Self {
                self.ln_1p()
            }

            fn exp(self) -> Self {
                self.exp()
            }

            fn exp2(self) -> Self {
                self.exp2()
            }

            fn exp10(self) -> Self {
                libm::Libm::<Self>::exp10(self)
            }

            fn exp_m1(self) -> Self {
                self.exp_m1()
            }

            fn floor(self) -> Self {
                self.floor()
            }

            fn ceil(self) -> Self {
                self.ceil()
            }

            fn round(self) -> Self {
                self.round()
            }

            fn trunc(self) -> Self {
                self.trunc()
            }

            fn frac(self) -> Self {
                self.fract()
            }

            fn abs(self) -> Self {
                self.abs()
            }

            fn sign(self) -> Self {
                match self.partial_cmp(&0.0) {
                    Some(cmp) => match cmp {
                        std::cmp::Ordering::Less => -1.0,
                        std::cmp::Ordering::Equal => 0.0,
                        std::cmp::Ordering::Greater => 1.0,
                    },
                    None => self,
                }
            }

            fn sqrt(self) -> Self {
                self.sqrt()
            }

            fn cbrt(self) -> Self {
                self.cbrt()
            }

            fn max(values: &[Self]) -> Self {
                values.iter().copied().fold(Self::MIN, Self::max)
            }

            fn min(values: &[Self]) -> Self {
                values.iter().copied().fold(Self::MAX, Self::min)
            }

            fn lgamma(self) -> Self {
                libm::Libm::<Self>::lgamma(self)
            }

            fn gamma(self) -> Self {
                libm::Libm::<Self>::tgamma(self)
            }

            fn factorial(self) -> Self {
                libm::Libm::<Self>::tgamma(self + 1.0)
            }

            fn double_factorial(self) -> Self {
                if self.is_infinite() || self < 0.0 || self.is_nan() {
                    return Self::NAN;
                }
                if self >= 301.0 {
                    return Self::INFINITY;
                }
                let mut result = 1.0;
                let mut k = self as u32;
                while k > 1 {
                    result *= k as $t;
                    k -= 2;
                }
                result
            }
        }
    };
}

impl_primitive!(f64);
impl_primitive!(f32);
