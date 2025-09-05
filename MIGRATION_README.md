# Database Migration untuk Token Usage System

## 🎯 Apa yang dilakukan?

Migration ini menambahkan table baru `user_monthly_token_usage` yang berisi pre-computed monthly statistics untuk performa yang jauh lebih cepat.

## 📋 Langkah-langkah Migration:

### 1. **Auto-Migration (Otomatis)**
Table baru akan dibuat otomatis saat aplikasi dijalankan:
```bash
python main.py
```
SQLAlchemy akan otomatis membuat table `user_monthly_token_usage` berdasarkan model yang sudah didefinisikan.

### 2. **Data Population (Manual)**
Untuk user yang sudah ada, jalankan script migration untuk populate historical data:
```bash
python migrate_token_data.py
```

Script ini akan:
- ✅ Scan semua user yang punya data token usage
- ✅ Aggregate data per bulan untuk setiap user  
- ✅ Populate table `user_monthly_token_usage`
- ✅ Show progress dan statistics

### 3. **Verifikasi**
Setelah migration, test API endpoint:
```bash
# Test dengan curl atau Postman
GET /auth/me
Authorization: Bearer <your-token>
```

Response harus include `token_stats` dengan data monthly:
```json
{
  "token_stats": {
    "monthly_stats": [
      {
        "month": "2024-12",
        "input_tokens": 15000,
        "output_tokens": 8500,
        "total_tokens": 23500,
        "total_cost": 0.45,
        "message_count": 25
      }
    ],
    "total_months": 1,
    "has_more": false
  }
}
```

## 🔧 Troubleshooting

### Error: "Table already exists"
Jika ada error table sudah ada, itu normal. SQLAlchemy skip table yang sudah ada.

### Error: "No token data found"
Jika tidak ada data token, berarti:
- Belum ada user yang melakukan chat dengan AI
- Token usage belum ditrack di messages table
- Normal untuk fresh database

### Error: Import issues
Pastikan jalankan dari root directory project:
```bash
cd /Users/shafiq/VsCodeProjects/foundation-be/
python migrate_token_data.py
```

## 📊 Performance Improvement

Setelah migration:
- **Sebelum**: 2-5 detik untuk aggregate data (lambat)
- **Sesudah**: 5-50ms untuk read pre-computed data (super cepat!)

## 🔄 Maintenance

System akan otomatis update monthly stats untuk:
- **Current month**: Update real-time setiap ada message baru
- **Historical months**: Sudah fix, tidak berubah

Tidak perlu maintenance manual! 🎉