import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import requests
import json
import threading
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import random
from collections import deque

class SensorFilter:
    def __init__(self, window_size=3):  # Daha küçük pencere, daha hızlı tepki
        self.window_size = window_size
        self.sensor_buffers = [deque(maxlen=window_size) for _ in range(4)]
        self.encoder_buffer = deque(maxlen=window_size)
        
    def filter_sensors(self, raw_sensors):
        """Sensör verilerini filtrele"""
        filtered = []
        for i, value in enumerate(raw_sensors):
            # Anormal değerleri filtrele
            if value < 2 or value > 400:  # Geçersiz aralık
                # Önceki değeri kullan veya varsayılan
                if len(self.sensor_buffers[i]) > 0:
                    value = self.sensor_buffers[i][-1]
                else:
                    value = 100  # Varsayılan uzak mesafe
            
            # Buffer'a ekle
            self.sensor_buffers[i].append(value)
            
            # Medyan filtre kullan (gürültüye karşı daha dayanıklı)
            if len(self.sensor_buffers[i]) >= 2:
                sorted_values = sorted(list(self.sensor_buffers[i]))
                filtered_value = sorted_values[len(sorted_values)//2]
            else:
                filtered_value = value
                
            filtered.append(filtered_value)
        
        return filtered
    
    def filter_encoder(self, raw_encoder):
        """Encoder verisini filtrele"""
        self.encoder_buffer.append(raw_encoder)
        
        # Ani sıçramaları filtrele
        if len(self.encoder_buffer) >= 2:
            diff = abs(raw_encoder - self.encoder_buffer[-2])
            if diff > 50:  # Çok büyük sıçrama
                return self.encoder_buffer[-2]  # Önceki değeri döndür
        
        return raw_encoder

class FastSLAM:
    def __init__(self, num_particles=30):  # Partikül sayısını azalttık
        self.num_particles = num_particles
        self.particles = []
        self.landmarks = []
        self.robot_path = []
        
        # Partikül filtreleri için parametreler
        self.motion_noise_distance = 0.005  # Azaltıldı
        self.motion_noise_theta = 0.02      # Azaltıldı
        self.sensor_noise = 0.05            # Azaltıldı
        
        # Hareket kararlılığı için
        self.consecutive_straight_moves = 0
        self.straight_move_threshold = 3
        
        # Partikülleri başlat
        for _ in range(num_particles):
            particle = {
                'x': 0.0,
                'y': 0.0,
                'theta': 0.0,
                'weight': 1.0 / num_particles,
                'landmarks': []
            }
            self.particles.append(particle)
    
    def predict(self, distance, theta_change):
        """Hareket modeli - daha kararlı tahmin"""
        # Çok küçük hareketleri yok say
        if distance < 0.005:
            return
            
        # Düz hareket algılama
        if abs(theta_change) < 0.05:  # Çok küçük dönüş
            self.consecutive_straight_moves += 1
            if self.consecutive_straight_moves >= self.straight_move_threshold:
                theta_change = 0  # Düz hareket olarak zorla
        else:
            self.consecutive_straight_moves = 0
        
        for particle in self.particles:
            # Düz hareket durumunda daha az gürültü
            if abs(theta_change) < 0.05:
                noisy_distance = distance + np.random.normal(0, self.motion_noise_distance * 0.3)
                noisy_theta = np.random.normal(0, self.motion_noise_theta * 0.1)
            else:
                noisy_distance = distance + np.random.normal(0, self.motion_noise_distance)
                noisy_theta = theta_change + np.random.normal(0, self.motion_noise_theta)
            
            # Yeni pozisyon hesapla
            particle['x'] += noisy_distance * math.cos(particle['theta'])
            particle['y'] += noisy_distance * math.sin(particle['theta'])
            particle['theta'] += noisy_theta
            
            # Açıyı normalize et
            particle['theta'] = math.atan2(math.sin(particle['theta']), math.cos(particle['theta']))
    
    def update(self, sensor_readings):
        """Sensör verisi ile partikülleri güncelle - daha kararlı"""
        total_weight = 0
        
        for particle in self.particles:
            weight = 1.0
            
            # Sadece geçerli sensör verilerini işle
            for i, distance in enumerate(sensor_readings):
                if 5 <= distance <= 50:  # Daha dar aralık, daha güvenilir veriler
                    # Sensör açısı (0: ön, 1: arka, 2: sağ, 3: sol)
                    sensor_angles = [0, math.pi, math.pi/2, -math.pi/2]
                    sensor_angle = sensor_angles[i]
                    
                    # Global koordinatlarda landmark pozisyonu
                    lm_x = particle['x'] + (distance/100) * math.cos(particle['theta'] + sensor_angle)
                    lm_y = particle['y'] + (distance/100) * math.sin(particle['theta'] + sensor_angle)
                    
                    # Yakın landmark ara - daha geniş tolerans
                    found_landmark = False
                    for lm in particle['landmarks']:
                        dist_to_lm = math.sqrt((lm['x'] - lm_x)**2 + (lm['y'] - lm_y)**2)
                        if dist_to_lm < 0.15:  # 15cm yakınlık
                            # Mevcut landmark'ı güncelle - daha yumuşak
                            alpha = 0.3  # Güncelleme hızı
                            lm['x'] = (1-alpha) * lm['x'] + alpha * lm_x
                            lm['y'] = (1-alpha) * lm['y'] + alpha * lm_y
                            weight *= 0.95  # Yüksek eşleşme ağırlığı
                            found_landmark = True
                            break
                    
                    if not found_landmark:
                        # Yeni landmark ekle - daha seçici
                        particle['landmarks'].append({'x': lm_x, 'y': lm_y})
                        weight *= 0.8  # Yeni landmark ağırlığı
            
            particle['weight'] = weight
            total_weight += weight
        
        # Ağırlıkları normalize et
        if total_weight > 0:
            for particle in self.particles:
                particle['weight'] /= total_weight
    
    def resample(self):
        """Ağırlık tabanlı yeniden örnekleme - daha az agresif"""
        weights = [p['weight'] for p in self.particles]
        
        # Effective sample size kontrolü
        ess = 1.0 / sum(w*w for w in weights)
        
        # Sadece gerektiğinde resample et
        if ess < self.num_particles * 0.5:
            indices = np.random.choice(len(self.particles), size=len(self.particles), p=weights)
            
            new_particles = []
            for idx in indices:
                new_particle = {
                    'x': self.particles[idx]['x'],
                    'y': self.particles[idx]['y'], 
                    'theta': self.particles[idx]['theta'],
                    'weight': 1.0 / self.num_particles,
                    'landmarks': [lm.copy() for lm in self.particles[idx]['landmarks']]
                }
                new_particles.append(new_particle)
            
            self.particles = new_particles
    
    def get_best_estimate(self):
        """En iyi tahmin - ağırlıklı ortalama"""
        total_weight = sum(p['weight'] for p in self.particles)
        if total_weight == 0:
            return 0, 0, 0
            
        # Ağırlıklı ortalama hesapla
        avg_x = sum(p['x'] * p['weight'] for p in self.particles) / total_weight
        avg_y = sum(p['y'] * p['weight'] for p in self.particles) / total_weight
        
        # Açı için özel hesaplama
        sin_sum = sum(math.sin(p['theta']) * p['weight'] for p in self.particles) / total_weight
        cos_sum = sum(math.cos(p['theta']) * p['weight'] for p in self.particles) / total_weight
        avg_theta = math.atan2(sin_sum, cos_sum)
        
        return avg_x, avg_y, avg_theta
    
    def get_all_landmarks(self):
        """Tüm partiküllerden landmark listesi oluştur - kümeleme ile"""
        all_landmarks = []
        for particle in self.particles:
            all_landmarks.extend(particle['landmarks'])
        
        # Yakın landmark'ları kümeleme
        clustered = []
        for lm in all_landmarks:
            found_cluster = False
            for cluster in clustered:
                dist = math.sqrt((cluster['x'] - lm['x'])**2 + (cluster['y'] - lm['y'])**2)
                if dist < 0.1:  # 10cm yakınlık
                    # Ortalama al
                    cluster['x'] = (cluster['x'] + lm['x']) / 2
                    cluster['y'] = (cluster['y'] + lm['y']) / 2
                    found_cluster = True
                    break
            
            if not found_cluster:
                clustered.append({'x': lm['x'], 'y': lm['y']})
        
        return clustered

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32 Robot SLAM Haritalama")
        self.root.geometry("1200x800")
        
        self.robot_ip = ""
        self.connected = False
        self.mapping_active = False
        self.data_thread = None
        
        # Sensör filtresi ekle
        self.sensor_filter = SensorFilter(window_size=3)
        
        # SLAM nesnesi
        self.slam = FastSLAM(num_particles=30)  # Azaltıldı
        self.last_encoder = 0
        self.last_theta = 0
        
        # Harita verileri
        self.robot_x = 0
        self.robot_y = 0
        self.robot_theta = 0
        
        # Hareket kararlılığı için
        self.position_history = deque(maxlen=5)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Ana çerçeve
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel - Kontroller
        left_frame = ttk.LabelFrame(main_frame, text="Robot Kontrolü", width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        left_frame.pack_propagate(False)
        
        # IP Bağlantısı
        conn_frame = ttk.LabelFrame(left_frame, text="WiFi Bağlantısı")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(conn_frame, text="Robot IP:").pack(anchor=tk.W)
        self.ip_entry = ttk.Entry(conn_frame, width=25)
        self.ip_entry.pack(fill=tk.X, padx=5, pady=2)
        self.ip_entry.insert(0, "172.20.10.11")  # Varsayılan IP - ESP32'nın IP adresini buraya girin
        
        ttk.Button(conn_frame, text="Bağlan", command=self.connect_robot).pack(pady=5)
        
        self.status_label = ttk.Label(conn_frame, text="Bağlı değil", foreground="red")
        self.status_label.pack(pady=2)

        # Sensör Verileri
        sensor_frame = ttk.LabelFrame(left_frame, text="Sensör Verileri")
        sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sensor_labels = []
        sensor_names = ["Ön", "Arka", "Sağ", "Sol"]
        for i, name in enumerate(sensor_names):
            label = ttk.Label(sensor_frame, text=f"{name}: -- cm")
            label.pack(anchor=tk.W, padx=5)
            self.sensor_labels.append(label)
        
        self.encoder_label = ttk.Label(sensor_frame, text="Encoder: --")
        self.encoder_label.pack(anchor=tk.W, padx=5)
        
        # Haritalama Kontrolleri
        map_frame = ttk.LabelFrame(left_frame, text="SLAM Haritalama")
        map_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(map_frame, text="Haritalamayı Başlat", command=self.start_mapping).pack(pady=2)
        ttk.Button(map_frame, text="Haritalamayı Durdur", command=self.stop_mapping).pack(pady=2)
        ttk.Button(map_frame, text="Haritayı Temizle", command=self.clear_map).pack(pady=2)
        ttk.Button(map_frame, text="Haritayı Kaydet", command=self.save_map).pack(pady=2)
        
        self.mapping_status = ttk.Label(map_frame, text="Haritalama: Kapalı", foreground="red")
        self.mapping_status.pack(pady=2)
        
        # Sağ panel - Harita görünümü
        right_frame = ttk.LabelFrame(main_frame, text="SLAM Haritası")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas oluştur
        self.canvas = tk.Canvas(right_frame, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Harita çizim değişkenleri
        self.canvas_width = 800
        self.canvas_height = 600
        self.scale = 100  # piksel/metre
        self.offset_x = self.canvas_width // 2
        self.offset_y = self.canvas_height // 2
        
        self.draw_grid()
        
    def connect_robot(self):
        """Robot ile WiFi bağlantısı kur"""
        self.robot_ip = self.ip_entry.get().strip()
        if not self.robot_ip:
            messagebox.showerror("Hata", "IP adresi girin!")
            return
        
        try:
            response = requests.get(f"http://{self.robot_ip}/status", timeout=3)
            if response.status_code == 200:
                self.connected = True
                self.status_label.config(text="Bağlandı", foreground="green")
                messagebox.showinfo("Başarılı", "Robot ile bağlantı kuruldu!")
            else:
                raise Exception("Bağlantı hatası")
        except Exception as e:
            self.connected = False
            self.status_label.config(text="Bağlı değil", foreground="red")
            messagebox.showerror("Hata", f"Bağlantı kurulamadı: {e}")

    def start_mapping(self):
        """SLAM haritalamayı başlat"""
        if not self.connected:
            messagebox.showerror("Hata", "Önce robot ile bağlantı kurun!")
            return

        # Haritalama durumunu kontrol et - eğer zaten aktifse durdur
        if self.mapping_active:
            self.stop_mapping()
            time.sleep(0.1)  # Kısa bekleme
        
        # Tüm verileri sıfırla - her başlatmada temiz başlangıç
        self.slam = FastSLAM(num_particles=30)
        self.last_encoder = 0
        self.last_theta = 0
        
        # Robot pozisyonunu merkeze sıfırla
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # Robot yolunu sıfırla ve başlangıç pozisyonunu ekle
        self.slam.robot_path = [(0.0, 0.0)]
        
        # Haritayı temizle ve grid'i yeniden çiz
        self.canvas.delete("all")
        self.draw_grid()
        
        # Haritalamayı aktif et
        self.mapping_active = True
        self.mapping_status.config(text="Haritalama: Aktif", foreground="green")
        
        # Encoder referansını al (başlangıç değeri)
        try:
            response = requests.get(f"http://{self.robot_ip}/sensors", timeout=2)
            if response.status_code == 200:
                data = response.json()
                self.last_encoder = data.get('encoder', 0)  # Başlangıç encoder değeri
        except:
            self.last_encoder = 0
        
        # Haritayı hemen güncelle (robot merkez pozisyonda görünsün)
        self.update_map()
                
                # Veri toplama thread'ini başlat
        self.data_thread = threading.Thread(target=self.data_collection_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        print("Haritalama sıfırlandı ve merkez pozisyondan başlatıldı")

    def stop_mapping(self):
        """SLAM haritalamayı durdur"""
        self.mapping_active = False
        self.mapping_status.config(text="Haritalama: Kapalı", foreground="red")
    
    def clear_map(self):
        """Haritayı temizle"""
        self.slam = FastSLAM(num_particles=30)
        self.last_encoder = 0
        self.last_theta = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.slam.robot_path = [(0.0, 0.0)]  # Başlangıç pozisyonunu ekle
        self.canvas.delete("all")
        self.draw_grid()
        self.update_map()  # Robotu hemen göster
    
    def data_collection_loop(self):
        """Veri toplama döngüsü"""
        while self.mapping_active and self.connected:
            try:
                # Sensör verilerini al
                response = requests.get(f"http://{self.robot_ip}/sensors", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Verileri güncelle
                    sensors = data.get('sensors', [0, 0, 0, 0])
                    encoder = data.get('encoder', 0)
                    
                    # GUI'yi güncelle
                    self.root.after(0, self.update_sensor_display, sensors, encoder)
                    
                    # SLAM işlemlerini yap
                    self.process_slam_data(sensors, encoder)
                    
                    # Haritayı güncelle
                    self.root.after(0, self.update_map)
                    
            except Exception as e:
                print(f"Veri alma hatası: {e}")
                
            time.sleep(0.2)  # 200ms bekleme
    
    def update_sensor_display(self, sensors, encoder):
        """Sensör verilerini ekranda güncelle"""
        sensor_names = ["Ön", "Arka", "Sağ", "Sol"]
        for i, (name, distance) in enumerate(zip(sensor_names, sensors)):
            self.sensor_labels[i].config(text=f"{name}: {distance:.1f} cm")
        
        self.encoder_label.config(text=f"Encoder: {encoder}")
    
    def process_slam_data(self, sensors, encoder):
        """SLAM algoritması için veri işle - gelişmiş dönüş algılama"""
        # Sensör verilerini filtrele
        filtered_sensors = self.sensor_filter.filter_sensors(sensors)
        filtered_encoder = self.sensor_filter.filter_encoder(encoder)
        
        # İlk veri alımında encoder referansını ayarla
        if self.last_encoder == 0:
            self.last_encoder = filtered_encoder
            print(f"Encoder referans değeri ayarlandı: {filtered_encoder}")
            return
        
        # Encoder'dan mesafe ve yön değişimi hesapla
        encoder_diff = filtered_encoder - self.last_encoder
        
        # Encoder sıfırlandıysa (overflow durumu)
        if encoder_diff < -1000:
            encoder_diff = filtered_encoder
            print(f"Encoder overflow algılandı, yeni referans: {filtered_encoder}")
            
        distance_moved = abs(encoder_diff) * 0.002  # Encoder pulse başına mesafe (metre)
        
        # Encoder yönü - pozitif ileri, negatif geri
        encoder_direction = 1 if encoder_diff > 0 else -1 if encoder_diff < 0 else 0
        
        # Gelişmiş theta değişimi hesaplaması
        theta_change = 0
        movement_detected = False
        
        if len(filtered_sensors) >= 4:
            front_dist = filtered_sensors[0]
            back_dist = filtered_sensors[1] 
            right_dist = filtered_sensors[2]
            left_dist = filtered_sensors[3]
            
            # Sensör geçmişini kontrol et (dönüş algılama için)
            if hasattr(self, 'last_sensors'):
                last_front, last_back, last_right, last_left = self.last_sensors
                
                # Sensör değişimleri
                front_change = front_dist - last_front
                back_change = back_dist - last_back
                right_change = right_dist - last_right
                left_change = left_dist - last_left
                
                # Dönüş algılama - yan sensörlerin değişim patterns
                side_diff_change = (right_change - left_change)
                
                # Çok belirgin dönüş patterns
                if abs(side_diff_change) > 8:  # 8cm'den fazla fark değişimi
                    if side_diff_change > 8:
                        # Sağ sensör sol sensörden çok daha fazla değişti - SOLA DÖNÜŞ
                        theta_change = 0.3 * encoder_direction  # Büyük açı değişimi
                        movement_detected = True
                        print(f"SOLA DÖNÜŞ algılandı - Side diff change: {side_diff_change:.1f}")
                    elif side_diff_change < -8:
                        # Sol sensör sağ sensörden çok daha fazla değişti - SAĞA DÖNÜŞ  
                        theta_change = -0.3 * encoder_direction
                        movement_detected = True
                        print(f"SAĞA DÖNÜŞ algılandı - Side diff change: {side_diff_change:.1f}")
                
                # Orta düzey dönüş algılama
                elif abs(side_diff_change) > 4:
                    if side_diff_change > 4:
                        theta_change = 0.15 * encoder_direction
                        movement_detected = True
                        print(f"Hafif sola dönüş - Side diff: {side_diff_change:.1f}")
                    elif side_diff_change < -4:
                        theta_change = -0.15 * encoder_direction  
                        movement_detected = True
                        print(f"Hafif sağa dönüş - Side diff: {side_diff_change:.1f}")
                
                # Ön/arka sensör değişimi ile dönüş teyidi
                if abs(front_change) > 10 or abs(back_change) > 10:
                    # Ön sensör çok değişti - muhtemelen engele yaklaşıp dönüyor
                    if front_change < -10:  # Öne engel yaklaştı
                        if left_dist > right_dist + 5:
                            theta_change += 0.2 * encoder_direction  # Sola dön
                            print(f"Ön engel - sola dönüş eklendi")
                        elif right_dist > left_dist + 5:
                            theta_change -= 0.2 * encoder_direction  # Sağa dön
                            print(f"Ön engel - sağa dönüş eklendi")
                        movement_detected = True
            
            # Mevcut sensör durumu ile anlık dönüş algılama
            current_side_diff = right_dist - left_dist
            if abs(current_side_diff) > 15:  # Büyük yan fark
                if current_side_diff > 15:
                    # Sağ taraf çok açık - sağa dönme eğilimi
                    theta_change += -0.1 * encoder_direction
                    movement_detected = True
                elif current_side_diff < -15:
                    # Sol taraf çok açık - sola dönme eğilimi  
                    theta_change += 0.1 * encoder_direction
                    movement_detected = True
            
            # Sensör geçmişini güncelle
            self.last_sensors = filtered_sensors.copy()
        else:
            # İlk kez sensör verisi - geçmiş yok
            self.last_sensors = filtered_sensors.copy()
        
        # Theta değişimini sınırla ama daha geniş aralık
        theta_change = np.clip(theta_change, -0.5, 0.5)
        
        # Her zaman sensör güncelleme yap
        self.slam.update(filtered_sensors)
        
        # SLAM güncelleme - daha düşük eşik, dönüşleri kaçırma
        if distance_moved > 0.002 or abs(theta_change) > 0.05 or movement_detected:
            self.slam.predict(distance_moved, theta_change)
            self.slam.resample()
            
            # Robot pozisyonunu güncelle
            new_x, new_y, new_theta = self.slam.get_best_estimate()
            
            # Pozisyon kararlılığı kontrolü - daha gevşek
            self.position_history.append((new_x, new_y, new_theta))
            
            if len(self.position_history) >= 3:
                # Ani sıçramaları filtrele ama dönüşlere izin ver
                recent_positions = list(self.position_history)[-3:]
                avg_x = sum(p[0] for p in recent_positions) / len(recent_positions)
                avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
                
                # Çok büyük sıçrama varsa (20cm+) önceki pozisyonu kullan
                if abs(new_x - avg_x) > 0.2 or abs(new_y - avg_y) > 0.2:
                    new_x = avg_x
                    new_y = avg_y
                    print(f"Pozisyon sıçraması filtrelendi")
            
            self.robot_x = new_x
            self.robot_y = new_y
            self.robot_theta = new_theta
            
            # Yola sadece anlamlı hareket varsa ekle - daha düşük eşik
            if len(self.slam.robot_path) == 0 or \
               math.sqrt((new_x - self.slam.robot_path[-1][0])**2 + 
                        (new_y - self.slam.robot_path[-1][1])**2) > 0.005:
                self.slam.robot_path.append((new_x, new_y))
            
            # Hareket tipi belirleme
            movement_type = "DÜZHATEKET"
            if abs(theta_change) > 0.2:
                movement_type = "SOLA DÖNÜŞ" if theta_change > 0 else "SAĞA DÖNÜŞ"
            elif abs(theta_change) > 0.1:
                movement_type = "HAFİF DÖNÜŞ"
            
            print(f"{movement_type} - Distance: {distance_moved:.4f}m, Theta: {theta_change:.3f}, Robot: ({new_x:.3f}, {new_y:.3f}), Açı: {math.degrees(new_theta):.1f}°")
        else:
            # Hareket olmasa bile mevcut en iyi tahmini al
            self.robot_x, self.robot_y, self.robot_theta = self.slam.get_best_estimate()
        
        self.last_encoder = filtered_encoder
        self.last_theta = self.robot_theta
    
    def draw_grid(self):
        """Harita grid'ini çiz"""
        # Dikey çizgiler
        for i in range(0, self.canvas_width, 50):
            self.canvas.create_line(i, 0, i, self.canvas_height, fill="lightgray", tags="grid")
        
        # Yatay çizgiler  
        for i in range(0, self.canvas_height, 50):
            self.canvas.create_line(0, i, self.canvas_width, i, fill="lightgray", tags="grid")
        
        # Merkez çizgileri
        self.canvas.create_line(self.offset_x, 0, self.offset_x, self.canvas_height, fill="gray", width=2)
        self.canvas.create_line(0, self.offset_y, self.canvas_width, self.offset_y, fill="gray", width=2)
    
    def update_map(self):
        """Haritayı güncelle"""
        # Eski harita elemanlarını temizle (grid hariç)
        self.canvas.delete("map")
        
        # Debug bilgisi ekle
        debug_text = f"Robot: ({self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_theta:.2f}°)"
        self.canvas.create_text(10, 10, text=debug_text, anchor="nw", fill="black", tags="map", font=("Arial", 10))
        
        # Robot yolunu çiz
        if len(self.slam.robot_path) > 1:
            points = []
            for x, y in self.slam.robot_path:
                screen_x = self.offset_x + x * self.scale
                screen_y = self.offset_y - y * self.scale  # Y eksenini ters çevir
                points.extend([screen_x, screen_y])
            
            if len(points) >= 4:
                self.canvas.create_line(points, fill="blue", width=3, tags="map")
        
        # Landmark'ları çiz
        landmarks = self.slam.get_all_landmarks()
        landmark_count = 0
        for lm in landmarks:
            screen_x = self.offset_x + lm['x'] * self.scale
            screen_y = self.offset_y - lm['y'] * self.scale
            # Landmark'ları daha büyük yap
            self.canvas.create_oval(screen_x-5, screen_y-5, screen_x+5, screen_y+5, 
                                  fill="red", outline="darkred", width=2, tags="map")
            landmark_count += 1
        
        # Landmark sayısını göster
        landmark_text = f"Landmark sayısı: {landmark_count}"
        self.canvas.create_text(10, 30, text=landmark_text, anchor="nw", fill="black", tags="map", font=("Arial", 10))
        
        # Robot pozisyonunu çiz - daha büyük ve görünür
        robot_screen_x = self.offset_x + self.robot_x * self.scale
        robot_screen_y = self.offset_y - self.robot_y * self.scale
        
        # Robot gövdesi - daha büyük
        robot_size = 15
        self.canvas.create_oval(robot_screen_x-robot_size, robot_screen_y-robot_size, 
                               robot_screen_x+robot_size, robot_screen_y+robot_size,
                               fill="green", outline="darkgreen", width=3, tags="map")
        
        # Robot merkez noktası
        self.canvas.create_oval(robot_screen_x-3, robot_screen_y-3, 
                               robot_screen_x+3, robot_screen_y+3,
                               fill="yellow", outline="orange", width=2, tags="map")
        
        # Robot yönü - daha uzun ok
        arrow_length = 25
        end_x = robot_screen_x + arrow_length * math.cos(self.robot_theta)
        end_y = robot_screen_y - arrow_length * math.sin(self.robot_theta)
        self.canvas.create_line(robot_screen_x, robot_screen_y, end_x, end_y,
                               fill="darkgreen", width=4, tags="map", arrow=tk.LAST, arrowshape=(16, 20, 6))
        
        # Eğer robot hareket etmiyorsa, başlangıç pozisyonunda göster
        if len(self.slam.robot_path) == 0:
            # Merkez pozisyonda statik robot göster
            center_x = self.offset_x
            center_y = self.offset_y
            self.canvas.create_oval(center_x-robot_size, center_y-robot_size, 
                                   center_x+robot_size, center_y+robot_size,
                                   fill="lightgreen", outline="green", width=3, tags="map")
            self.canvas.create_text(center_x, center_y+robot_size+10, text="Robot (Başlangıç)", 
                                  anchor="center", fill="green", tags="map", font=("Arial", 10, "bold"))
        
        # Path sayısını göster
        path_text = f"Yol noktası sayısı: {len(self.slam.robot_path)}"
        self.canvas.create_text(10, 50, text=path_text, anchor="nw", fill="black", tags="map", font=("Arial", 10))
    
    def save_map(self):
        """Haritayı PNG olarak kaydet"""
        if not self.slam.robot_path:
            messagebox.showwarning("Uyarı", "Kaydedilecek harita verisi yok!")
            return
        
        try:
            # Yüksek çözünürlüklü görüntü oluştur
            img_width, img_height = 1200, 900
            img = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Siyah çerçeve çiz
            draw.rectangle([0, 0, img_width-1, img_height-1], outline='black', width=3)
            
            # Harita boyutlarını hesapla
            all_x = [pos[0] for pos in self.slam.robot_path]
            all_y = [pos[1] for pos in self.slam.robot_path]
            obstacles = self.slam.get_all_landmarks()
            
            if obstacles:
                all_x.extend([lm['x'] for lm in obstacles])
                all_y.extend([lm['y'] for lm in obstacles])
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                # Marjin ekle
                margin = 0.5
                min_x -= margin
                max_x += margin
                min_y -= margin
                max_y += margin
                
                # Ölçek hesapla
                width_scale = (img_width - 100) / (max_x - min_x) if max_x != min_x else 100
                height_scale = (img_height - 100) / (max_y - min_y) if max_y != min_y else 100
                scale = min(width_scale, height_scale)
                
                # Offset hesapla
                offset_x = 50 + (img_width - 100 - (max_x - min_x) * scale) / 2
                offset_y = 50 + (img_height - 100 - (max_y - min_y) * scale) / 2
                
                # Robot yolunu çiz
                if len(self.slam.robot_path) > 1:
                    path_points = []
                    for x, y in self.slam.robot_path:
                        screen_x = offset_x + (x - min_x) * scale
                        screen_y = img_height - (offset_y + (y - min_y) * scale)
                        path_points.append((screen_x, screen_y))
                    
                    if len(path_points) > 1:
                        draw.line(path_points, fill='blue', width=3)
                
                # Landmark'ları çiz (engeller)
                landmark_points = []
                for lm in obstacles:
                    screen_x = offset_x + (lm['x'] - min_x) * scale
                    screen_y = img_height - (offset_y + (lm['y'] - min_y) * scale)
                    landmark_points.append((screen_x, screen_y))
                    draw.ellipse([screen_x-4, screen_y-4, screen_x+4, screen_y+4], 
                               fill='red', outline='darkred')
                
                # Robot son pozisyon
                if self.slam.robot_path:
                    last_x, last_y = self.slam.robot_path[-1]
                    robot_screen_x = offset_x + (last_x - min_x) * scale
                    robot_screen_y = img_height - (offset_y + (last_y - min_y) * scale)
                    
                    draw.ellipse([robot_screen_x-8, robot_screen_y-8,
                                robot_screen_x+8, robot_screen_y+8],
                               fill='green', outline='darkgreen')
            
            # Dosyayı kaydet
            filename = f"robot_map_{int(time.time())}.png"
            img.save(filename)
            messagebox.showinfo("Başarılı", f"Harita kaydedildi: {filename}")
                
        except Exception as e:
            messagebox.showerror("Hata", f"Harita kaydedilemedi: {e}")

def main():
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 