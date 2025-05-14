#include <DHT.h>

#define DHTPIN 25       // Pin digital yang terhubung ke sensor DHT22
#define DHTTYPE DHT22  // Tipe sensor DHT22 (AM2302)

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
}

void loop() {
  delay(2000);  // Tunggu 2 detik antara pembacaan

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();  // Baca suhu dalam Celsius

  // Periksa apakah pembacaan berhasil
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Gagal membaca dari sensor DHT22!");
  } else {
    Serial.print("Kelembaban: ");
    Serial.print(humidity);
    Serial.print(" %\t");
    Serial.print("Suhu: ");
    Serial.print(temperature);
    Serial.println(" *C");
  }
}
