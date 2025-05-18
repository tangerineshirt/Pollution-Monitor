#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Ganti sesuai WiFi kamu
const char* ssid = "BuzzNet 2.4Ghz_EXT";
const char* password = "BuzzNet6988";

// Ganti IP server kamu, pastikan port 80 aktif dan endpoint tersedia
const char* serverName = "http://52.200.111.58/api/sensor";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Menghubungkan ke WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nTerkoneksi ke WiFi!");

  kirimDataHTTP();
}

void loop() {
  // Tidak melakukan apa-apa di loop
}

void kirimDataHTTP() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);  // Langsung tanpa SSL
    http.addHeader("Content-Type", "application/json");

    // Data JSON dummy
    String jsonData = "{\"pm25\":42.1,\"co\":0.65,\"no2\":18.9,\"temp\":26.7,\"humidity\":59.3,\"air_quality\":\"Good\"}";

    int httpResponseCode = http.POST(jsonData);

    Serial.print("Kode respon HTTP: ");
    Serial.println(httpResponseCode);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("Isi respon dari server:");
      Serial.println(response);

      // Parse JSON dari response
      DynamicJsonDocument doc(1024);
      DeserializationError error = deserializeJson(doc, response);

      if (!error) {
        const char* status = doc["status"];
        const char* message = doc["message"];

        Serial.print("Status dari JSON: ");
        Serial.println(status);

        Serial.print("Pesan dari JSON: ");
        Serial.println(message);
      } else {
        Serial.print("Gagal parse JSON: ");
        Serial.println(error.c_str());
      }

    } else {
      Serial.print("Gagal kirim data. Error: ");
      Serial.println(http.errorToString(httpResponseCode).c_str());
    }

    http.end();
  } else {
    Serial.println("WiFi tidak terkoneksi.");
  }
}
