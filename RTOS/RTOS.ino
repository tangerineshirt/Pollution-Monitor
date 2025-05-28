//Nama: Razzan Naufal Rianta
//NIM: 225150300111037
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

SemaphoreHandle_t xSemaphore = NULL;

void TaskSensor(void *pvParameters);
void TaskInference(void *pvParameters);

int dataset[10];
int index_data = 0;
bool dataSiap = false;

void setup() {
  Serial.begin(115200);
  xSemaphore = xSemaphoreCreateMutex();
  if (xSemaphore != NULL) {
    xTaskCreate(TaskSensor, "Sensor", 4096, NULL, 3, NULL);
    xTaskCreate(TaskInference, "Inference", 4096, NULL, 2, NULL);
  }
}

void loop() {}

void TaskSensor(void *pvParameters) {
  for (;;) {
    vTaskDelay(100 / portTICK_PERIOD_MS);
    if (xSemaphoreTake(xSemaphore, (TickType_t)5) == pdTRUE) {
      int data = random(1, 1000);
      dataset[index_data] = data;
      Serial.print("Data ke ");
      Serial.print(index_data + 1);
      Serial.print(": ");
      Serial.println(data);
      index_data++;
      if (index_data >= 10) {
        index_data = 0;
        dataSiap = true;
      }
      xSemaphoreGive(xSemaphore);
    }
  }
}

void TaskInference(void *pvParameters) {
  for (;;) {
    vTaskDelay(1000 / portTICK_PERIOD_MS);
    if (xSemaphoreTake(xSemaphore, (TickType_t)5) == pdTRUE) {
      if (dataSiap) {
        int total = 0;
        for (int i = 0; i < 10; i++) {
          total += dataset[i];
        }
        Serial.print("Hasil inferensi: ");
        Serial.println(total);
      } else {
        Serial.println("Belum ada cukup data.");
      }
      xSemaphoreGive(xSemaphore);
    }
  }
}
