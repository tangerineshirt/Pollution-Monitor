// #include <tflm_esp32.h>
// #include <eloquent_tinyml.h>
#include <Arduino.h>
#include <GP2YDustSensor.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <DHT.h>
#include <WiFi.h>
#include <HTTPClient.h>
// #include "airQuality.h"

#define NO2 25  // Pin analog input ESP32
#define DHTPIN 19
#define DHTTYPE DHT22

const char *ssid = "BuzzNet 5Ghz";
const char *password = "BuzzNet6988";

const char *serverName = "http://52.200.111.58/api/sensor";

// WiFiClientSecure client;

SemaphoreHandle_t xSemaphore;
DHT dht(DHTPIN, DHTTYPE);

const uint8_t SHARP_LED_PIN = 14;  // Sharp Dust/particle sensor Led Pin
const uint8_t SHARP_VO_PIN = A0;   // Sharp Dust/particle analog out pin used for reading

// Eloquent::TF::Sequential<TF_NUM_OPS, 2000> tf;
GP2YDustSensor dustSensor(GP2YDustSensorType::GP2Y1010AU0F, SHARP_LED_PIN, SHARP_VO_PIN);

void TaskSensor(void *pvParameters);
// void TaskPredict(void *pvParameters);
void TaskSendData(void *pvParameters);
void TaskPrint(void *pvParameters);

float humidity = 0, temp = 0, pm25 = 0, co = 0, no2 = 0;
int currentTask = 0;

const int mq7Pin = 35;    // Gunakan pin ADC1 (GPIO 32-39) untuk pembacaan
const float RLmq = 10.0;  // Nilai resistor beban (load resistor) dalam kilo ohm
float Ro = 10.0;          // Ro harus dikalibrasi di udara bersih (estimasi awal)
int mqValue = 0;
float mqVoltage = 0, Rs = 0, ratio = 0, ppm_log = 0, ppm = 0;
const float mq_b = 0.35, mq_m = -0.77;

const float VREF = 3.3;      // Tegangan referensi ESP32
const float RLno = 47000.0;  // Resistor beban 47kΩ

float no2concentration = 0;

int noadc = 0;
float noVoltage = 0, rs = 0, ppb = 0, log_rs = 0, log_ppb = 0;
const float no_m = -0.7, no_b = 3.3;

float dense = 0;
float running = 0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  // WiFi.begin(ssid, password);

  dht.begin();
  dustSensor.begin();
  // tf.begin(airQualityModel);
  if (xSemaphore == NULL) {
    xSemaphore = xSemaphoreCreateMutex();
    xSemaphoreGive((xSemaphore));
    xTaskCreate(TaskSensor, "Sensor", 8192, NULL, 3, NULL);
    // xTaskCreate(TaskPredict, "Predict", 4096, NULL, 2, NULL);
    xTaskCreate(TaskSendData, "SendData", 2048, NULL, 2, NULL);
    xTaskCreate(TaskPrint, "Print", 2048, NULL, 1, NULL);
  }
  // client.setInsecure();
}

void loop() {
  // kosong
}

void TaskSensor(void *pvParameters) {
  for (;;) {
    //INI DHT22
    humidity = dht.readHumidity();
    temp = dht.readTemperature();

    //INI MQ-7
    mqValue = analogRead(mq7Pin);                 // 0 - 4095 (12-bit ADC)
    mqVoltage = mqValue * (3.3 / 4095.0);         // Tegangan aktual (ESP32 3.3V)
    Rs = ((3.3 - mqVoltage) / mqVoltage) * RLmq;  // Rumus resistansi sensor saat ini
    ratio = Rs / Ro;

    // Kurva dari datasheet MQ7 (CO): log(PPM) = (log(Rs/Ro) - b) / m
    ppm_log = (log10(ratio) - mq_b) / mq_m;
    ppm = pow(10, ppm_log);  // Konversi log PPM ke nilai PPM asli

    //INI NO2
    noadc = analogRead(NO2);
    noVoltage = noadc * (3.3 / 4096);
    no2concentration = noVoltage * 2.0;

    //INI PM2.5
    dense = dustSensor.getDustDensity();
    running = dustSensor.getRunningAverage();
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void TaskSendData(void *pvParameters) {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println("Connecting to WiFi...");
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
  Serial.println("Connected to WiFi");

  for (;;) {
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;

      // Siapkan data JSON
      String postData = "{";
      postData += "\"pm25\":" + String(dense, 2) + ",";
      postData += "\"co\":" + String(ppm, 2) + ",";
      postData += "\"no2\":" + String(no2concentration, 2) + ",";
      postData += "\"temp\":" + String(temp, 2) + ",";
      postData += "\"humidity\":" + String(humidity, 2) + ",";
      postData += "\"air_quality\":\"Good\"";  // Jika punya perhitungan AQI, ganti ini
      postData += "}";

      http.begin(serverName);
      http.addHeader("Content-Type", "application/json");

      int httpResponseCode = http.POST(postData);

      if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        Serial.print("Response: ");
        Serial.println(response);
      } else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);
      }

      http.end();
    } else {
      Serial.println("WiFi Disconnected. Attempting reconnect...");
      WiFi.begin(ssid, password);
    }

    vTaskDelay(10000 / portTICK_PERIOD_MS);  // kirim tiap 10 detik
  }
}

// void TaskPredict(void *pvParameters) {
//   for (;;) {
//     float input[5] = {temp, humidity, pm25, no2, co};
//     float output[4];

//     tf.predict(input, output);

//     int predicted_class = 0;
//     float max_score = output[0];
//     for (int i = 1; i < 4; i++) {
//       if (output[i] > max_score) {
//         max_score = output[i];
//         predicted_class = i;
//       }
//     }

//     Serial.print("Predicted Class: ");
//     switch (predicted_class) {
//       case 0:
//         Serial.println("Hazardous");
//         break;
//       case 1:
//         Serial.println("Poor");
//         break;
//       case 2:
//         Serial.println("Moderate");
//         break;
//       case 3:
//         Serial.println("Good");
//         break;
//       default:
//         Serial.println("Unknown");
//         break;
//     }

//     vTaskDelay(2100 / portTICK_PERIOD_MS);
//   }
// }

void TaskPrint(void *pvParameters) {
  for (;;) {
    Serial.print("Suhu: ");
    Serial.println(temp);
    Serial.print("Kelembaban: ");
    Serial.println(humidity);
    Serial.print("CO: ");
    Serial.println(ppm);
    Serial.print("NO2: ");
    Serial.println(no2concentration);
    Serial.print("PM2.5: ");
    Serial.print(dense);
    Serial.println(" ug/m3");
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}
