#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ================= WiFi Ayarları =================
const char* ssid = "Furkan";
const char* password = "furkan123";

// ================= Web Sunucusu =================
WebServer server(80);

// ================= Arka Motor Sürücü 1 Pinleri =================
const int IN1_1 = 16;
const int IN2_1 = 17;
const int IN3_1 = 18;
const int IN4_1 = 19;

// ================= Ön Motor Sürücü 2 Pinleri =================
const int IN1_2 = 27;
const int IN2_2 = 26;
const int IN3_2 = 25;
const int IN4_2 = 33;

// ================= Encoder Ayarları =================
const int encoderPin = 32;
volatile long encoderCount = 0;
const float wheelDiameter = 0.068; // metre cinsinden 6.8 cm
const int pulsesPerRevolution = 40;

// ================= HC-SR04 Sensör Pinleri =================
#define NUM_SENSORS 4
const int trigPins[NUM_SENSORS] = {23, 4, 22, 21};
const int echoPins[NUM_SENSORS] = {14, 12, 5, 13}; // 0: Ön, 1: Arka, 2: Sağ, 3: Sol

long durations[NUM_SENSORS];
float distances[NUM_SENSORS];

// ================= Encoder Interrupt =================
void IRAM_ATTR encoderISR() {
  encoderCount++;
}

// ================= Motor Kontrol Fonksiyonları =================
void ileriGit() {
  digitalWrite(IN1_1, HIGH); digitalWrite(IN2_1, LOW);
  digitalWrite(IN3_1, HIGH); digitalWrite(IN4_1, LOW);
  digitalWrite(IN1_2, HIGH); digitalWrite(IN2_2, LOW);
  digitalWrite(IN3_2, HIGH); digitalWrite(IN4_2, LOW);
}

void geriGit() {
  digitalWrite(IN1_1, LOW); digitalWrite(IN2_1, HIGH);
  digitalWrite(IN3_1, LOW); digitalWrite(IN4_1, HIGH);
  digitalWrite(IN1_2, LOW); digitalWrite(IN2_2, HIGH);
  digitalWrite(IN3_2, LOW); digitalWrite(IN4_2, HIGH);
}

void sagaDon() {
  // Sol motorlar geri, sağ motorlar ileri
  digitalWrite(IN1_1, LOW); digitalWrite(IN2_1, HIGH);
  digitalWrite(IN3_1, HIGH); digitalWrite(IN4_1, LOW);
  digitalWrite(IN1_2, HIGH); digitalWrite(IN2_2, LOW);
  digitalWrite(IN3_2, LOW); digitalWrite(IN4_2, HIGH);
}

void solaDon() {
  // Sol motorlar ileri, sağ motorlar geri
  digitalWrite(IN1_1, HIGH); digitalWrite(IN2_1, LOW);
  digitalWrite(IN3_1, LOW); digitalWrite(IN4_1, HIGH);
  digitalWrite(IN1_2, LOW); digitalWrite(IN2_2, HIGH);
  digitalWrite(IN3_2, HIGH); digitalWrite(IN4_2, LOW);
}

void dur() {
  digitalWrite(IN1_1, LOW); digitalWrite(IN2_1, LOW);
  digitalWrite(IN3_1, LOW); digitalWrite(IN4_1, LOW);
  digitalWrite(IN1_2, LOW); digitalWrite(IN2_2, LOW);
  digitalWrite(IN3_2, LOW); digitalWrite(IN4_2, LOW);
}

// ================= Web Server Fonksiyonları =================
void handleRoot() {
  String html = "<html><body>";
  html += "<h1>ESP32 Robot Kontrol</h1>";
  html += "<p>Robot aktif ve çalışıyor.</p>";
  html += "<p><a href='/status'>Durum</a> | <a href='/sensors'>Sensörler</a></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleStatus() {
  DynamicJsonDocument doc(1024);
  doc["status"] = "active";
  doc["encoder"] = encoderCount;
  doc["wifi_connected"] = WiFi.status() == WL_CONNECTED;
  doc["ip"] = WiFi.localIP().toString();
  
  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}

void handleSensors() {
  DynamicJsonDocument doc(1024);
  JsonArray sensors = doc.createNestedArray("sensors");
  
  sensors.add(distances[0]); // Ön
  sensors.add(distances[1]); // Arka
  sensors.add(distances[2]); // Sağ
  sensors.add(distances[3]); // Sol
  
  doc["encoder"] = encoderCount;
  doc["timestamp"] = millis();
  
  String response;
  serializeJson(doc, response);
  server.send(200, "application/json", response);
}

void handleNotFound() {
  server.send(404, "text/plain", "Sayfa bulunamadı");
}

void setupWebServer() {
  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.on("/sensors", handleSensors);
  server.onNotFound(handleNotFound);
  server.begin();
  Serial.println("Web sunucusu başlatıldı");
}

// ================= Kurulum =================
void setup() {
  Serial.begin(115200);

  // WiFi bağlantısı
  WiFi.begin(ssid, password);
  Serial.print("WiFi'ye bağlanıyor");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.println("WiFi bağlandı!");
  Serial.print("IP Adresi: ");
  Serial.println(WiFi.localIP());

  pinMode(IN1_1, OUTPUT); pinMode(IN2_1, OUTPUT);
  pinMode(IN3_1, OUTPUT); pinMode(IN4_1, OUTPUT);
  pinMode(IN1_2, OUTPUT); pinMode(IN2_2, OUTPUT);
  pinMode(IN3_2, OUTPUT); pinMode(IN4_2, OUTPUT);

  pinMode(encoderPin, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(encoderPin), encoderISR, RISING);

  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
  }

  ileriGit(); // Başlangıçta ileri git
  
  // Web sunucusunu başlat
  setupWebServer();
}

// ================= Ana Döngü =================
void loop() {
  // Encoder mesafesi ölçümü
  noInterrupts();
  long count = encoderCount;
  interrupts();

  float revolutions = (float)count / pulsesPerRevolution;
  float distanceTraveled = revolutions * (3.1416 * wheelDiameter);
  Serial.print("Pulse: "); Serial.print(count);
  Serial.print(" | Tur: "); Serial.print(revolutions, 4);
  Serial.print(" | Mesafe (m): "); Serial.println(distanceTraveled, 4);

  // Sensörleri tetikle ve ölç
  for (int i = 0; i < NUM_SENSORS; i++) {
    digitalWrite(trigPins[i], LOW);
    delayMicroseconds(2);
    digitalWrite(trigPins[i], HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPins[i], LOW);

    durations[i] = pulseIn(echoPins[i], HIGH, 25000);
    if (durations[i] == 0) {
      distances[i] = 1000.0;  // Ölçüm yoksa uzak kabul et
    } else {
      distances[i] = durations[i] * 0.034 / 2;
    }
    Serial.print("Sensor "); Serial.print(i);
    Serial.print(": "); Serial.print(distances[i]);
    Serial.println(" cm");
  }

  float front = distances[0];
  float back = distances[1];
  float right = distances[2];
  float left = distances[3];

  if (front < 35) {
    Serial.println("Önde engel! Geri git.");
    geriGit();
    delay(600);

    if (right > left) {
      Serial.println("Engel sağda daha az, sağa dön.");
      sagaDon();
      delay(600);
    } else {
      Serial.println("Engel solda daha az, sola dön.");
      solaDon();
      delay(600);
    }

  } else if (right < 30) {
    Serial.println("Engel sağda, sola dön.");
    solaDon();
    delay(600);

  } else if (left < 30) {
    Serial.println("Engel solda, sağa dön.");
    sagaDon();
    delay(600);

  } else {
    Serial.println("Yol açık, ileri git.");
    ileriGit();
  }

  delay(200);
  
  // Web server isteklerini işle
  server.handleClient();
}
