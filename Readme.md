# RAG (Retrieval Augmented Generation) Soru-Cevap Sistemi

Bu proje, belgelerden bilgi çıkarımı yapan ve kullanıcı sorularına bağlam duyarlı cevaplar üreten bir RAG (Retrieval Augmented Generation) sistemidir.

## Özellikler

- PDF, Jupyter Notebook ve metin dosyalarını işleyebilme
- Bağlam duyarlı soru-cevap
- Oturum bazlı geçmiş yönetimi
- Benzer soruları tespit etme ve önceki cevapları kullanma
- Quiz oluşturma özelliği

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. OpenAI API anahtarınızı ayarlayın:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Kullanım

1. Sunucuyu başlatın:
```bash
python rag_server.py
```

2. Yeni bir terminal açın ve istemciyi başlatın:
```bash
python ask_question.py
```

3. Kullanılabilir komutlar:
- `q` veya `question`: Soru sor
- `h` veya `history`: Geçmiş soruları göster
- `s` veya `suggestions`: Önerilen soruları göster
- `a` veya `add`: Yeni soru ekle
- `u` veya `upload`: Dosya yükle
- `l` veya `load`: Klasörden dosya yükle
- `z` veya `quiz`: Quiz oluştur
- `exit`: Çıkış

## Proje Yapısı

- `rag_server.py`: FastAPI tabanlı RAG sunucusu
- `ask_question.py`: Soru-cevap istemcisi
- `key_words.py`: Alan bazlı anahtar kelimeler
- `dokumanlar/`: Belgelerin bulunduğu klasör

## Gereksinimler

- Python 3.8+
- FastAPI
- LangChain
- OpenAI
- FAISS
- Sentence Transformers
- PyPDF2
- nbformat

## Lisans

MIT