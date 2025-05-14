// Pin setup
const int mq7Pin = 25; // gunakan salah satu pin ADC ESP32 (GPIO34, GPIO35, dsb)


void setup() {
  Serial.begin(115200);
}

void loop() {
  int adcValue = analogRead(mq7Pin);

  Serial.print("ADC Value: ");
  Serial.println(adcValue);

  delay(1000); // Update setiap 1 detik
}