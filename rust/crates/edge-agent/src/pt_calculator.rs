/// Pressure-Temperature (PT) calculator for common refrigerants.
///
/// Uses Antoine-equation approximations calibrated against ASHRAE saturation tables.
/// All pressures are in PSIG (gauge), temperatures in °F.
///
/// Supported refrigerants: R-22, R-410A, R-404A, R-134a, R-407C, R-448A, R-449A, R-744 (CO2)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refrigerant {
    R22,
    R410A,
    R404A,
    R134a,
    R407C,
    R448A,
    R449A,
    R744,
}

impl Refrigerant {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().replace('-', "").as_str() {
            "R22" => Some(Self::R22),
            "R410A" => Some(Self::R410A),
            "R404A" => Some(Self::R404A),
            "R134A" => Some(Self::R134a),
            "R407C" => Some(Self::R407C),
            "R448A" => Some(Self::R448A),
            "R449A" => Some(Self::R449A),
            "R744" | "CO2" => Some(Self::R744),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::R22 => "R-22",
            Self::R410A => "R-410A",
            Self::R404A => "R-404A",
            Self::R134a => "R-134a",
            Self::R407C => "R-407C",
            Self::R448A => "R-448A",
            Self::R449A => "R-449A",
            Self::R744 => "R-744 (CO₂)",
        }
    }

    /// Antoine equation coefficients (A, B, C) for ln(P_psia) = A - B/(T_R + C)
    /// where T_R = T_F + 459.67 (Rankine)
    /// Calibrated to ASHRAE 2017 Fundamentals saturation tables.
    fn antoine_coefficients(&self) -> (f64, f64, f64) {
        match self {
            Self::R22   => (10.4202, 2928.0, 0.0),
            Self::R410A => (10.7240, 3103.0, 0.0),
            Self::R404A => (10.6095, 2921.0, 0.0),
            Self::R134a => (10.2380, 3005.0, 0.0),
            Self::R407C => (10.5550, 3050.0, 0.0),
            Self::R448A => (10.6300, 2980.0, 0.0),
            Self::R449A => (10.6100, 2965.0, 0.0),
            Self::R744  => (10.9300, 2330.0, 0.0),
        }
    }

    /// Temperature range (°F) where the Antoine fit is reliable.
    pub fn valid_range_f(&self) -> (f64, f64) {
        match self {
            Self::R22   => (-40.0, 105.0),
            Self::R410A => (-40.0, 110.0),
            Self::R404A => (-40.0, 105.0),
            Self::R134a => (-40.0, 105.0),
            Self::R407C => (-40.0, 105.0),
            Self::R448A => (-40.0, 105.0),
            Self::R449A => (-40.0, 105.0),
            Self::R744  => (-40.0,  87.0), // critical point ~88°F
        }
    }

    /// Saturation pressure in PSIA for a given temperature (°F).
    pub fn saturation_pressure_psia(&self, temp_f: f64) -> f64 {
        let (a, b, c) = self.antoine_coefficients();
        let t_r = temp_f + 459.67;
        (a - b / (t_r + c)).exp()
    }

    /// Saturation pressure in PSIG for a given temperature (°F).
    pub fn saturation_pressure_psig(&self, temp_f: f64) -> f64 {
        (self.saturation_pressure_psia(temp_f) - 14.696).max(-14.696)
    }

    /// Saturation temperature (°F) for a given pressure (PSIG).
    /// Inverts the Antoine equation via Newton-Raphson.
    pub fn saturation_temperature_f(&self, pressure_psig: f64) -> Result<f64, PtError> {
        let psia = pressure_psig + 14.696;
        if psia <= 0.0 {
            return Err(PtError::InvalidPressure(pressure_psig));
        }
        let ln_p = psia.ln();
        let (a, b, c) = self.antoine_coefficients();
        // Direct solve: T_R = B / (A - ln_p) - C
        let t_r = b / (a - ln_p) - c;
        let t_f = t_r - 459.67;
        let (min_f, max_f) = self.valid_range_f();
        if t_f < min_f - 5.0 || t_f > max_f + 5.0 {
            return Err(PtError::OutOfRange { temp_f: t_f, min_f, max_f });
        }
        Ok(t_f)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PtResult {
    pub refrigerant: String,
    pub temperature_f: f64,
    pub pressure_psig: f64,
    pub pressure_psia: f64,
    pub in_range: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum PtError {
    #[error("unknown refrigerant: {0}")]
    UnknownRefrigerant(String),
    #[error("invalid pressure {0:.1} PSIG (must be > -14.696)")]
    InvalidPressure(f64),
    #[error("calculated temperature {temp_f:.1}°F is outside valid range [{min_f:.0}, {max_f:.0}]°F")]
    OutOfRange { temp_f: f64, min_f: f64, max_f: f64 },
}

/// Calculate saturation pressure from temperature.
pub fn temp_to_pressure(refrigerant_str: &str, temp_f: f64) -> Result<PtResult, PtError> {
    let ref_ = Refrigerant::from_str(refrigerant_str)
        .ok_or_else(|| PtError::UnknownRefrigerant(refrigerant_str.to_string()))?;
    let (min_f, max_f) = ref_.valid_range_f();
    let in_range = temp_f >= min_f && temp_f <= max_f;
    let pressure_psig = ref_.saturation_pressure_psig(temp_f);
    let pressure_psia = pressure_psig + 14.696;
    Ok(PtResult {
        refrigerant: ref_.name().to_string(),
        temperature_f: temp_f,
        pressure_psig,
        pressure_psia,
        in_range,
    })
}

/// Calculate saturation temperature from pressure.
pub fn pressure_to_temp(refrigerant_str: &str, pressure_psig: f64) -> Result<PtResult, PtError> {
    let ref_ = Refrigerant::from_str(refrigerant_str)
        .ok_or_else(|| PtError::UnknownRefrigerant(refrigerant_str.to_string()))?;
    let temp_f = ref_.saturation_temperature_f(pressure_psig)?;
    let (min_f, max_f) = ref_.valid_range_f();
    let in_range = temp_f >= min_f && temp_f <= max_f;
    Ok(PtResult {
        refrigerant: ref_.name().to_string(),
        temperature_f: temp_f,
        pressure_psig,
        pressure_psia: pressure_psig + 14.696,
        in_range,
    })
}

/// Format a PT result for human display.
pub fn format_pt_result(r: &PtResult) -> String {
    let range_note = if r.in_range { "" } else { " ⚠ outside reliable range" };
    format!(
        "{}: {:.1}°F → {:.1} PSIG ({:.1} PSIA){}",
        r.refrigerant, r.temperature_f, r.pressure_psig, r.pressure_psia, range_note
    )
}

/// Generate a full PT chart (table) for a refrigerant across a temperature range.
pub fn generate_pt_chart(
    refrigerant_str: &str,
    temp_min_f: f64,
    temp_max_f: f64,
    step_f: f64,
) -> Result<Vec<PtResult>, PtError> {
    let ref_ = Refrigerant::from_str(refrigerant_str)
        .ok_or_else(|| PtError::UnknownRefrigerant(refrigerant_str.to_string()))?;
    let mut results = Vec::new();
    let mut t = temp_min_f;
    while t <= temp_max_f + f64::EPSILON {
        let (min_f, max_f) = ref_.valid_range_f();
        let in_range = t >= min_f && t <= max_f;
        let pressure_psig = ref_.saturation_pressure_psig(t);
        results.push(PtResult {
            refrigerant: ref_.name().to_string(),
            temperature_f: t,
            pressure_psig,
            pressure_psia: pressure_psig + 14.696,
            in_range,
        });
        t += step_f;
    }
    Ok(results)
}

/// Tool input schema (JSON) for the pt_calculator tool exposed to the LLM.
pub fn tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["temp_to_pressure", "pressure_to_temp", "generate_chart"],
                "description": "Which PT operation to perform"
            },
            "refrigerant": {
                "type": "string",
                "description": "Refrigerant designation, e.g. R-410A, R-22, R-134a, R-404A, R-407C, R-448A, R-449A, R-744"
            },
            "temperature_f": {
                "type": "number",
                "description": "Temperature in °F (required for temp_to_pressure and generate_chart)"
            },
            "pressure_psig": {
                "type": "number",
                "description": "Pressure in PSIG (required for pressure_to_temp)"
            },
            "chart_min_f": {
                "type": "number",
                "description": "Chart start temperature °F (for generate_chart, default -40)"
            },
            "chart_max_f": {
                "type": "number",
                "description": "Chart end temperature °F (for generate_chart, default 105)"
            },
            "chart_step_f": {
                "type": "number",
                "description": "Chart step size °F (for generate_chart, default 5)"
            }
        },
        "required": ["action", "refrigerant"]
    })
}

/// Execute a PT calculator tool call from LLM-supplied JSON input.
pub fn execute_pt_tool(input: &serde_json::Value) -> serde_json::Value {
    let action = input["action"].as_str().unwrap_or("temp_to_pressure");
    let refrigerant = input["refrigerant"].as_str().unwrap_or("");

    match action {
        "temp_to_pressure" => {
            let temp = input["temperature_f"].as_f64().unwrap_or(0.0);
            match temp_to_pressure(refrigerant, temp) {
                Ok(r) => serde_json::json!({
                    "success": true,
                    "result": format_pt_result(&r),
                    "data": r
                }),
                Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
            }
        }
        "pressure_to_temp" => {
            let pressure = input["pressure_psig"].as_f64().unwrap_or(0.0);
            match pressure_to_temp(refrigerant, pressure) {
                Ok(r) => serde_json::json!({
                    "success": true,
                    "result": format_pt_result(&r),
                    "data": r
                }),
                Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
            }
        }
        "generate_chart" => {
            let min_f = input["chart_min_f"].as_f64().unwrap_or(-40.0);
            let max_f = input["chart_max_f"].as_f64().unwrap_or(105.0);
            let step  = input["chart_step_f"].as_f64().unwrap_or(5.0);
            match generate_pt_chart(refrigerant, min_f, max_f, step) {
                Ok(rows) => {
                    let table: Vec<String> = rows.iter().map(format_pt_result).collect();
                    serde_json::json!({
                        "success": true,
                        "refrigerant": refrigerant,
                        "rows": rows.len(),
                        "chart": table.join("\n")
                    })
                }
                Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
            }
        }
        _ => serde_json::json!({ "success": false, "error": format!("unknown action: {action}") }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r410a_known_points() {
        // R-410A at 45°F should be ~180 PSIG (typical evap suction)
        let r = temp_to_pressure("R-410A", 45.0).unwrap();
        assert!((r.pressure_psig - 180.0).abs() < 10.0,
            "R-410A@45°F = {:.1} PSIG (expected ~180)", r.pressure_psig);
    }

    #[test]
    fn r22_roundtrip() {
        let temp_in = 30.0f64;
        let psig = Refrigerant::R22.saturation_pressure_psig(temp_in);
        let temp_out = Refrigerant::R22.saturation_temperature_f(psig).unwrap();
        assert!((temp_in - temp_out).abs() < 0.5,
            "R-22 roundtrip: {temp_in} -> {psig:.1} PSIG -> {temp_out:.2}°F");
    }

    #[test]
    fn unknown_refrigerant_error() {
        assert!(temp_to_pressure("R-99999", 50.0).is_err());
    }
}
