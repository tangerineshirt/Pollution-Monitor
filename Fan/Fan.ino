#define ENA 27    // PWM pin
#define IN1 14
#define IN2 13

const int pwmFreq = 5000;         // 5kHz PWM
const int pwmResolution = 8;      // 8-bit: nilai antara 0-255

void setup() {
  Serial.begin(115200);
  Serial.println("Mulai test kipas...");

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Atur arah putaran kipas
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Setup PWM
  ledcAttach(ENA, pwmFreq, pwmResolution);

  // Jalankan kipas perlahan-lahan naikkan speed
  for (int speed = 0; speed <= 255; speed += 51) {
    ledcWrite(ENA, speed);
    Serial.print("Speed: ");
    Serial.println(speed);
    delay(1000);  // tunggu 1 detik per level
  }
  ledcWrite(ENA, 0);

  Serial.println("Uji kipas selesai.");
}

void loop() {
  // Tidak ada apa-apa di loop
}
