# Makine Öğrenmesi Final Ödevi — Müşteri Kaybı Tahmini

Bu proje, bir telekom müşterisinin hizmetten ayrılıp ayrılmayacağını (`Churn`) tahmin eden uçtan uca bir **ikili sınıflandırma** çalışmasıdır. Amaç yalnızca yüksek skor elde etmek değil; veri inceleme, ön işleme, öznitelik mühendisliği, model karşılaştırma, çapraz doğrulama, hiperparametre ayarlama ve açıklanabilirlik adımlarını temiz ve tekrar çalıştırılabilir bir yapıda göstermektir.

## Veri seti

Projede IBM'nin **Telco Customer Churn** örnek veri seti kullanılmıştır. Veri setinde 7.043 müşteri ve 21 ham sütun bulunur. Hedef değişken `Churn` olup `Yes` müşterinin ayrıldığını, `No` ise kaldığını gösterir.

- [IBM kaynak deposu](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [Kullanılan ham CSV](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

`TotalCharges` sütunundaki boş değerler sayısal forma dönüşüm sırasında eksik değer olarak ele alınır. Doldurma işlemleri veri sızıntısını önlemek amacıyla model Pipeline'ı içinde yalnızca eğitim verisine fit edilir.

## Uygulanan makine öğrenmesi akışı

1. Ham veri inceleme ve hedef dağılımı
2. Eksik değer kontrolü ve tip dönüşümü
3. Kategorik değişkenlerde one-hot encoding
4. Sayısal değişkenlerde medyan doldurma ve `RobustScaler`
5. IQR ile aykırı değer incelemesi
6. Dört yeni öznitelik üretimi:
   - `ServiceCount`
   - `AverageMonthlySpend`
   - `TenureGroup`
   - `HasSecuritySupport`
7. `VarianceThreshold` ile düşük varyanslı öznitelik seçimi
8. Stratify ile `%60 eğitim / %20 validation / %20 test` ayrımı
9. Logistic Regression, KNN ve Random Forest karşılaştırması
10. Beş katlı stratified çapraz doğrulama
11. Validation F1 lideri için Grid Search
12. Ayrı test kümesinde nihai değerlendirme
13. Permutation importance ile modelden bağımsız açıklanabilirlik

## Model karşılaştırması

Model seçiminin ana metriği, ayrılan müşterileri bulma ile yanlış alarm dengesini birlikte değerlendiren **F1-score** olarak belirlenmiştir.

| Model | CV F1 ortalama | Validation Accuracy | Validation Precision | Validation Recall | Validation F1 | Validation ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6246 | 0.7771 | 0.5628 | 0.7193 | **0.6315** | 0.8375 |
| Logistic Regression | **0.6318** | 0.7445 | 0.5124 | **0.7754** | 0.6170 | 0.8365 |
| KNN | 0.5490 | 0.7750 | 0.5893 | 0.5027 | 0.5426 | 0.7947 |

Validation F1 lideri **Random Forest** olmuştur. Grid Search sonucu:

```text
class_weight      = balanced
max_depth         = 8
min_samples_leaf  = 1
n_estimators      = 250
```

Beş katlı Grid Search en iyi CV F1 değeri: **0.6412**.

## Test sonucu

Validation ve ayarlama sürecinde kullanılmayan test kümesindeki sonuçlar:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.7693 |
| Precision | 0.5474 |
| Recall | **0.7567** |
| F1-score | **0.6352** |
| ROC-AUC | **0.8430** |

Model ayrılan müşterilerin yaklaşık `%75.7`sini yakalamaktadır. Precision'ın daha düşük olması, bazı kalacak müşterilerin de riskli işaretlendiğini gösterir. Müşteri kaybını kaçırmanın maliyetinin gereksiz bir elde tutma teklifinden yüksek olduğu senaryolarda bu recall seviyesi iş açısından anlamlı olabilir; gerçek kullanımda karar eşiği kampanya maliyetine göre ayrıca ayarlanmalıdır.

## Açıklanabilirlik ve sınırlılıklar

Bonus bölümünde permutation importance uygulanmıştır. Bu yöntem, test verisinde bir değişken karıştırıldığında F1 skorunun ne kadar düştüğünü ölçerek modelden bağımsız önem yorumu sağlar. Tam sıralama `outputs/permutation_importance.csv`, grafik ise `outputs/permutation_importance.png` dosyasındadır.

Başlıca sınırlılıklar:

- Veri tek bir örnek telekom müşteri tablosundan gelir; farklı dönem ve şirketlerde dış doğrulama yapılmamıştır.
- Hedef sınıf dengesizdir; accuracy tek başına yeterli değildir.
- Model nedensellik değil ilişki öğrenir. Bir değişkenin önemli olması, müşteri kaybına doğrudan neden olduğunu kanıtlamaz.
- İşlem maliyeti ve müşteri yaşam boyu değeri bulunmadığından karar eşiği finansal optimizasyonla seçilmemiştir.

## Proje yapısı

```text
makine_ogrenmesi_final_odevi/
├── data/
│   ├── README.md
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── Makine_Ogrenmesi_Final_Odevi.ipynb
├── outputs/
│   ├── best_model.joblib
│   ├── best_params.json
│   ├── confusion_matrix.png
│   ├── model_comparison.csv
│   ├── outlier_summary.csv
│   ├── permutation_importance.csv
│   ├── permutation_importance.png
│   └── test_metrics.json
├── src/
│   ├── __init__.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Kurulum ve çalıştırma

Python 3.10 veya daha güncel bir sürüm önerilir.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Notebook'u açmak için:

```bash
jupyter lab notebooks/Makine_Ogrenmesi_Final_Odevi.ipynb
```

Komut satırı çalışması tamamlandığında model, metrik tabloları ve grafikler otomatik olarak `outputs/` klasörüne yazılır.
