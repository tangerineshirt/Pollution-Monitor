#include <Arduino.h>
#include <GP2YDustSensor.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>
#include <DHT.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "SVMClassifier.h"

#define NO2 25  // Pin analog input ESP32
#define DHTPIN 21
#define DHTTYPE DHT22
#define DHTPIN2 22
#define DHTPIN3 23
#define ENA 27
#define IN1 14
#define IN2 13
#define MERAH 16
#define KUNING 17
#define PUTIH 18
#define HIJAU 19

SemaphoreHandle_t xSemaphore;
DHT dht(DHTPIN, DHTTYPE);
DHT dht2(DHTPIN2, DHTTYPE);
DHT dht3(DHTPIN3, DHTTYPE);

const char *ssid = "OPPO A77s";
const char *password = "password";
const char *serverUrl = "http://52.200.111.58/api/sensor";

const uint8_t SHARP_LED_PIN = 5;  // Sharp Dust/particle sensor Led Pin
const uint8_t SHARP_VO_PIN = 34;  // Sharp Dust/particle analog out pin used for reading

Eloquent::ML::Port::SVM clf;
GP2YDustSensor dustSensor(GP2YDustSensorType::GP2Y1010AU0F, SHARP_LED_PIN, SHARP_VO_PIN);

void TaskSensor(void *pvParameters);
void TaskPredict(void *pvParameters);
void TaskSendData(void *pvParameters);
void TaskPrint(void *pvParameters);

float humidity = 0, temp = 0, pm25 = 0, co = 0, no2 = 0;
int currentTask = 0;

const int mq7Pin = 35;  // Gunakan pin ADC1 (GPIO 32-39) untuk pembacaan
const int mq7pin2 = 36;
const int mq7pin3 = 39;
const float RLmq = 10.0;  // Nilai resistor beban (load resistor) dalam kilo ohm
float Ro = 10.0;          // Ro harus dikalibrasi di udara bersih (estimasi awal)
int mqValue = 0;
float mqVoltage = 0, Rs = 0, ratio = 0, ppm_log = 0;
const float mq_b = 0.35, mq_m = -0.77;

const float VREF = 3.3;      // Tegangan referensi ESP32
const float RLno = 47000.0;  // Resistor beban 47kΩ

String kelas;
int noadc = 0;
float noVoltage = 0;
float running = 0;

//DC FAN
const int pwmFreq = 5000;
const int pwmResolution = 8;  // 8-bit (0 - 255)
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(MERAH, OUTPUT);
  pinMode(KUNING, OUTPUT);
  pinMode(PUTIH, OUTPUT);
  pinMode(HIJAU, OUTPUT);
  ledcAttach(ENA, pwmFreq, pwmResolution);

  dht.begin();
  dht2.begin();
  dht3.begin();
  dustSensor.begin();
  if (xSemaphore == NULL) {
    xSemaphore = xSemaphoreCreateMutex();
    if (xSemaphore != NULL) {
      if (xTaskCreate(TaskSensor, "Sensor", 16384, NULL, 4, NULL) != pdPASS)
        Serial.println("Failed to create Sensor task");
      if (xTaskCreate(TaskPredict, "Predict", 16384, NULL, 3, NULL) != pdPASS)
        Serial.println("Failed to create Predict task");
      if (xTaskCreatePinnedToCore(TaskSendData, "SendData", 16384, NULL, 2, NULL, 0) != pdPASS)
        Serial.println("Failed to create SendData task");
      if (xTaskCreate(TaskPrint, "Print", 4096, NULL, 1, NULL) != pdPASS)
        Serial.println("Failed to create Print task");
    } else {
      Serial.println("Failed to create semaphore.");
    }
  }
}

void loop() {}

float getMax(float a, float b, float c) {
  float arr[3] = { a, b, c };
  // Bubble sort
  for (int i = 0; i < 2; i++) {
    for (int j = i + 1; j < 3; j++) {
      if (arr[i] > arr[j]) {
        float temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }
  return arr[2]; 
}

float calibrateCO(float x) {
  float ppm;
  mqVoltage = x * (3.3 / 4095.0);
  Rs = ((3.3 - mqVoltage) / mqVoltage) * RLmq;
  ratio = Rs / Ro;

  // Kurva dari datasheet MQ7 (CO): log(PPM) = (log(Rs/Ro) - b) / m
  ppm_log = (log10(ratio) - mq_b) / mq_m;
  ppm = pow(10, ppm_log);
  return ppm;
}

void TaskSensor(void *pvParameters) {
  for (;;) {
    float temp1 = 0;
    float temp2 = 0;
    float temp3 = 0;

    float hum1 = 0;
    float hum2 = 0;
    float hum3 = 0;

    float mq1 = 0;
    float mq2 = 0;
    float mq3 = 0;
    if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
      //INI DHT22
      temp1 = dht.readTemperature();
      temp2 = dht2.readTemperature();
      temp3 = dht3.readTemperature();

      hum1 = dht.readHumidity();
      hum2 = dht2.readHumidity();
      hum3 = dht3.readHumidity();

      temp = getMax(temp1, temp2, temp3);
      humidity = getMax(hum1, hum2, hum3);

      //INI MQ-7
      mq1 = calibrateCO(analogRead(mq7Pin));
      mq2 = calibrateCO(analogRead(mq7pin2));
      mq3 = calibrateCO(analogRead(mq7pin3));

      Serial.println(mq1);
      Serial.println(mq2);
      Serial.println(mq3);
      co = getMax(mq1, mq2, mq3);

      //INI NO2
      noadc = analogRead(NO2);
      noVoltage = noadc * (3.3 / 4096);
      no2 = noVoltage * 2.0;

      //INI PM2.5
      pm25 = dustSensor.getDustDensity();
      running = dustSensor.getRunningAverage();
      xSemaphoreGive(xSemaphore);
    }
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void TaskPredict(void *pvParameters) {
  for (;;) {
    if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
      int speed = 0;

      float input[5] = { temp, humidity, pm25, no2, co };

      int predictedClass = clf.predict(input);
      digitalWrite(MERAH, LOW);
      digitalWrite(KUNING, LOW);
      digitalWrite(PUTIH, LOW);
      digitalWrite(HIJAU, LOW);
      switch (predictedClass) {
        case 0:
          kelas = "Hazardous";
          speed = 255;
          digitalWrite(MERAH, HIGH);
          break;
        case 1:
          kelas = "Poor";
          speed = 255;
          digitalWrite(KUNING, HIGH);
          break;
        case 2:
          kelas = "Moderate";
          speed = 200;
          digitalWrite(PUTIH, HIGH);
          break;
        case 3:
          kelas = "Good";
          speed = 0;
          digitalWrite(HIJAU, HIGH);
          break;
        default:
          kelas = "Unknown";
          speed = 0;
          break;
      }
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);

      ledcWrite(ENA, speed);

      xSemaphoreGive(xSemaphore);
    }
    vTaskDelay(2100 / portTICK_PERIOD_MS);
  }
}

void TaskSendData(void *pvParameters) {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  for (;;) {
    if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {

      StaticJsonDocument<256> doc;
      doc["pm25"] = pm25;
      doc["co"] = co;
      doc["no2"] = no2;
      doc["temp"] = temp;
      doc["humidity"] = humidity;
      doc["air_quality"] = kelas;

      String jsonStr;
      serializeJson(doc, jsonStr);

      if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "application/json");
        int httpResponseCode = http.POST(jsonStr);
        http.end();

        if (httpResponseCode > 0) {
          Serial.println("Data sent to server.");
        } else {
          Serial.println("Send failed.");
        }
      } else {
        Serial.println("WiFi disconnected.");
      }
      xSemaphoreGive(xSemaphore);
    }
    vTaskDelay(10000 / portTICK_PERIOD_MS);  // setiap 10 detik
  }
}

void TaskPrint(void *pvParameters) {
  for (;;) {
    if (xSemaphoreTake(xSemaphore, portMAX_DELAY) == pdTRUE) {
      Serial.print("Suhu: ");
      Serial.println(temp);
      Serial.print("Kelembaban: ");
      Serial.println(humidity);
      Serial.print("CO: ");
      Serial.println(co);
      Serial.print("NO2: ");
      Serial.println(no2);
      Serial.print("PM2.5: ");
      Serial.print(pm25);
      Serial.println(" ug/m3");
      Serial.print("Predicted Class: ");
      Serial.println(kelas);
      xSemaphoreGive(xSemaphore);
    }
    vTaskDelay(2500 / portTICK_PERIOD_MS);
  }
}