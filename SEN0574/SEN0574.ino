#define SENSOR_PIN 25  // Pin analog input ESP32
const float VREF = 3.3;     // Tegangan referensi ESP32
const float RL = 47000.0;   // Resistor beban 47kΩ

void setup() {
  Serial.begin(115200);
}

void loop() {
  int adc = analogRead(SENSOR_PIN);
  float voltage = (adc / 4095.0) * VREF;

  float rs = 0;
  float ppb = 0;

  if (voltage >= 0.01) {  // Hindari divide-by-zero
    rs = RL * ((VREF - voltage) / voltage);

    float log_rs = log10(rs);

    // Persamaan kalibrasi (dari datasheet, aproksimasi)
    float m = -0.7;
    float b = 3.3;

    float log_ppb = (log_rs - b) / m;
    ppb = pow(10, log_ppb);
  }

  Serial.print("ADC: "); Serial.print(adc);
  Serial.print(" | Tegangan: "); Serial.print(voltage, 3); Serial.print(" V");
  Serial.print(" | Rs: "); Serial.print(rs, 1);
  Serial.print(" ohm | NO2: "); Serial.print(ppb, 1);
  Serial.println(" ppb");

  delay(1000);
}