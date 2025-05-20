import requests
import json
from datetime import datetime
import os
from typing import Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import aiohttp
from key_words import DOMAIN_KEYWORDS  # key_words.py'den anahtar kelimeleri içe aktar

class RAGClient:
    def __init__(self):
        self.url = "http://localhost:5001"
        self.history = []
        self.context = []  # Bağlam için yeni liste
        self.max_context_length = 5  # Maksimum bağlam uzunluğu
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # Oturum ID'si
        self.history_file = f"qa_history_{self.session_id}.json"  # Oturuma özel dosya adı
        self.similarity_threshold = 0.75  # Benzerlik eşiğini düşürdük
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.suggestions = {
            "Makine Öğrenmesi": [
                "Supervised learning nedir?",
                "Unsupervised learning nedir?",
                "Overfitting ve underfitting nedir?",
                "Cross-validation nedir?",
                "Regularization teknikleri nelerdir?",
                "Gradient descent nasıl çalışır?",
                "Random Forest algoritması nasıl çalışır?",
                "SVM (Support Vector Machine) nedir?",
                "K-means clustering nasıl çalışır?",
                "Neural Network mimarileri nelerdir?"
            ],
            "Derin Öğrenme": [
                "CNN (Convolutional Neural Network) nedir?",
                "RNN (Recurrent Neural Network) nedir?",
                "LSTM ve GRU arasındaki farklar nelerdir?",
                "Transfer learning nedir?",
                "Attention mekanizması nasıl çalışır?",
                "Batch normalization nedir?",
                "Dropout nedir ve neden kullanılır?",
                "Activation fonksiyonları nelerdir?",
                "Backpropagation nasıl çalışır?",
                "GAN (Generative Adversarial Network) nedir?"
            ],
            "Veri Analizi": [
                "Veri ön işleme adımları nelerdir?",
                "Feature engineering nedir?",
                "Feature selection teknikleri nelerdir?",
                "Dimensionality reduction nedir?",
                "PCA (Principal Component Analysis) nasıl çalışır?",
                "Outlier detection yöntemleri nelerdir?",
                "Imbalanced data nasıl ele alınır?",
                "Time series analizi nedir?",
                "A/B testing nedir?",
                "Confusion matrix nedir?"
            ]
        }
        
        # Oturum başlangıç mesajı
        print(f"\nYeni oturum başlatıldı (ID: {self.session_id})")
        print(f"Oturum geçmişi: {self.history_file}")
        
    def calculate_similarity(self, question1: str, question2: str) -> float:
        """İki soru arasındaki benzerlik skorunu hesapla"""
        # Soruları vektörlere dönüştür
        vec1 = self.model.encode([question1])[0]
        vec2 = self.model.encode([question2])[0]
        
        # Cosine similarity hesapla
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return float(similarity)
    
    def get_question_embedding(self, question: str) -> np.ndarray:
        """Soru için embedding vektörü oluştur"""
        return self.model.encode([question])[0]
    
    def clean_answer(self, answer: str) -> str:
        """Cevaptan kaynak önekini temizle"""
        prefixes = [
            "[OpenAI Genel Bilgi Tabanından] ",
            "[Yüklenen Dosyalardan] "
        ]
        for prefix in prefixes:
            if answer.startswith(prefix):
                return answer[len(prefix):]
        return answer
    
    def load_all_histories(self):
        """Tüm oturum geçmişlerini yükle"""
        all_histories = []
        for file in os.listdir():
            if file.startswith("qa_history_") and file.endswith(".json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                        # Sadece başarılı cevapları al
                        successful_qa = []
                        for qa in history:
                            if isinstance(qa, dict):
                                answer = qa.get('answer', '')
                                if not (answer.startswith("[OpenAI Genel Bilgi Tabanından] Üzgünüm") or 
                                      answer.startswith("[Yüklenen Dosyalardan]")):
                                    successful_qa.append(qa)
                        if successful_qa:
                            all_histories.extend(successful_qa)
                except Exception as e:
                    print(f"Uyarı: {file} dosyası okunamadı: {str(e)}")
        return all_histories
        
    def find_similar_question(self, question: str) -> Optional[tuple]:
        """Benzer bir soru var mı kontrol et"""
        # Tüm oturum geçmişlerini yükle
        all_histories = self.load_all_histories()
        
        if not all_histories:
            return None
            
        best_score = 0
        best_match = None
        
        # Tüm geçmiş soruları kontrol et
        for qa in all_histories:
            if isinstance(qa, dict):
                q = qa.get('question', '')
                a = qa.get('answer', '')
                
                if not q or not a:  # Boş soru veya cevap varsa atla
                    continue
                    
                # Benzerlik skorunu hesapla
                score = self.calculate_similarity(question, q)
                
                # Eğer benzerlik skoru eşik değerinden yüksekse ve önceki en iyi skordan daha iyiyse
                if score > self.similarity_threshold and score > best_score:
                    best_score = score
                    best_match = (q, a)
                    print(f"\nBenzer soru bulundu (Benzerlik: {score:.2f}):")
                    print(f"Soru: {q}")
                    print(f"Cevap: {a}")
        
        return best_match

    def update_context(self, question: str, answer: str):
        """Bağlamı güncelle"""
        # Eğer cevap başarısızsa veya veritabanından geldiyse bağlama ekleme
        if answer.startswith("[OpenAI Genel Bilgi Tabanından] Üzgünüm") or answer.startswith("[Yüklenen Dosyalardan]"):
            return
            
        # Sorunun teknik alanını belirle
        question_domain = None
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in question.lower() for keyword in keywords):
                question_domain = domain
                break
                
        # Mevcut bağlamdaki sorularla benzerlik kontrolü
        for context_item in self.context:
            similar_score = self.calculate_similarity(question, context_item['question'])
            if similar_score > 0.9:  # Çok benzer sorular için eşiği yükselttik
                # Benzer soru bulundu, bağlama ekleme
                return
                
        # Yeni soru-cevap çiftini bağlama ekle
        self.context.append({
            "question": question,
            "answer": answer,
            "domain": question_domain
        })
        
        # Bağlam uzunluğunu kontrol et
        if len(self.context) > self.max_context_length:
            self.context.pop(0)  # En eski bağlamı çıkar

    def get_context_string(self) -> str:
        """Bağlamı string olarak döndür"""
        if not self.context:
            return ""
            
        context_str = "Önceki konuşma bağlamı:\n"
        for i, item in enumerate(self.context, 1):
            context_str += f"{i}. Soru: {item['question']}\n"
            context_str += f"   Cevap: {item['answer']}\n"
        return context_str + "\n"

    async def ask_question(self, question: str) -> str:
        """Soruyu sor ve cevabı al"""
        try:
            # Benzer soru kontrolü
            similar = self.find_similar_question(question)
            if similar:
                similar_q, similar_a = similar
                print(f"\nBenzer bir soru bulundu (Benzerlik: {self.calculate_similarity(question, similar_q):.2f}):")
                print(f"Soru: {similar_q}")
                print(f"Cevap: {similar_a}")
                
                # Eğer soru alanı belirliyse ve benzer soru farklı bir alandaysa, yeni cevap iste
                question_domain = None
                similar_domain = None
                
                for domain, keywords in DOMAIN_KEYWORDS.items():
                    if any(keyword in question.lower() for keyword in keywords):
                        question_domain = domain
                    if any(keyword in similar_q.lower() for keyword in keywords):
                        similar_domain = domain
                
                if question_domain and similar_domain and question_domain != similar_domain:
                    print("\nBu soru farklı bir teknik alanla ilgili. Yeni bir cevap almak ister misiniz? (e/h)")
                    if input().lower() == 'e':
                        answer = await self.get_new_answer(question)
                        self.update_context(question, answer)
                        return answer
                    self.update_context(question, similar_a)
                    return similar_a
                
                # Benzerlik skoru yüksekse direkt kullan
                if self.calculate_similarity(question, similar_q) > 0.9:
                    self.update_context(question, similar_a)
                    return similar_a
                
                # Benzerlik skoru düşükse kullanıcıya sor
                print("\nBu cevabı kullanmak ister misiniz? (e/h)")
                if input().lower() == 'e':
                    self.update_context(question, similar_a)
                    return similar_a
            
            # Benzer soru bulunamadıysa veya kullanıcı yeni cevap istiyorsa
            answer = await self.get_new_answer(question)
            self.update_context(question, answer)
            return answer
            
        except Exception as e:
            print(f"\nHata: {str(e)}")
            return f"Üzgünüm, bir hata oluştu: {str(e)}"
            
    async def get_new_answer(self, question: str) -> str:
        """Yeni bir cevap al"""
        try:
            # Bağlamı ekle
            context_str = ""
            if self.context:
                # Sorunun domain'ini belirle
                question_domain = None
                for domain, keywords in DOMAIN_KEYWORDS.items():
                    if any(keyword in question.lower() for keyword in keywords):
                        question_domain = domain
                        break
                
                # Bağlamdaki soruları filtrele
                context_questions = []
                related_questions = []
                
                for item in self.context:
                    if question_domain and item.get('domain') == question_domain:
                        context_questions.append(item)
                    else:
                        # İki soru arasındaki benzerliği kontrol et
                        similarity = self.calculate_similarity(question, item['question'])
                        if similarity > 0.5:  # Daha düşük bir eşik kullanıyoruz
                            related_questions.append(item)
                
                # Bağlam string'ini oluştur
                if context_questions:
                    context_str += "Önceki konuşma bağlamı:\n"
                    for i, item in enumerate(context_questions, 1):
                        context_str += f"{i}. Soru: {item['question']}\n"
                        context_str += f"   Cevap: {item['answer']}\n"
                    context_str += "\n"
                
                if related_questions:
                    context_str += "İlgili önceki konuşmalar:\n"
                    for i, item in enumerate(related_questions, 1):
                        context_str += f"{i}. Soru: {item['question']}\n"
                        context_str += f"   Cevap: {item['answer']}\n"
                    context_str += "\n"
                
                # Eğer soru önceki bir soruya referans veriyorsa (örn: "hangileri", "bunlar", "onlar" gibi)
                if any(word in question.lower() for word in ["hangileri", "bunlar", "onlar", "en", "daha"]):
                    context_str += "Not: Bu soru önceki soruya referans veriyor. Lütfen önceki bağlamı dikkate alarak cevap verin.\n\n"
                
                question = f"{context_str}Yeni soru: {question}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:5001/query",
                    json={"question": question}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("answer", "")
                        
                        # Cevabı temizle
                        answer = self.clean_answer(answer)
                        
                        # Eğer cevap veritabanından geliyorsa kaydetme
                        if answer.startswith("[Yüklenen Dosyalardan]"):
                            print("\nBu cevap veritabanından geldiği için kaydedilmeyecek.")
                            return answer
                            
                        # Başarısız cevapları kaydetme
                        if answer.startswith("Üzgünüm"):
                            print("\nBu cevap başarısız olduğu için kaydedilmeyecek.")
                            return answer
                            
                        # Cevabı kaydet
                        self.save_to_file(question, answer)
                        return answer
                    else:
                        error_msg = await response.text()
                        return f"Üzgünüm, bir hata oluştu: {error_msg}"
        except Exception as e:
            return f"Üzgünüm, bir hata oluştu: {str(e)}"
    
    def upload_file(self, file_path: str) -> dict:
        """Yeni bir dosya yükle"""
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            response = requests.post(f"{self.url}/upload", files=files)
            response.raise_for_status()
            return response.json()
    
    def load_files(self, directory: str) -> dict:
        """Belirtilen klasördeki tüm desteklenen dosyaları yükle"""
        response = requests.post(f"{self.url}/load-files", params={"directory": directory})
        response.raise_for_status()
        return response.json()
        
    def save_to_file(self, question: str, answer: str, source_session_id: Optional[str] = None) -> None:
        """Soru ve cevabı oturuma özel dosyaya kaydet"""
        # Eğer cevap başarısızsa veya veri tabanından geldiyse kaydetme
        if answer.startswith("[OpenAI Genel Bilgi Tabanından] Üzgünüm") or answer.startswith("[Yüklenen Dosyalardan]"):
            if answer.startswith("[Yüklenen Dosyalardan]"):
                print("\nBu cevap kaydedilmeyecek çünkü veri tabanından geldi.")
            else:
                print("\nBu cevap kaydedilmeyecek çünkü yeterli bilgi bulunamadı.")
            return
            
        # Eğer cevap aynı oturumdan geliyorsa kaydetme
        if source_session_id == self.session_id:
            print("\nBu cevap kaydedilmeyecek çünkü aynı oturumdan geldi.")
            return
            
        # Cevabı temizle
        clean_answer = self.clean_answer(answer)
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": clean_answer,
            "session_id": self.session_id
        }
        self.history.append(entry)
        
        # Dosyaya kaydet
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def show_history(self) -> None:
        """Soru-cevap geçmişini göster"""
        if not self.history:
            print("\nBu oturumda henüz soru-cevap geçmişi yok.")
            return
            
        print(f"\nOturum ID: {self.session_id}")
        print("Soru-Cevap Geçmişi:")
        print("-" * 50)
        for entry in self.history:
            print(f"\nZaman: {entry['timestamp']}")
            print(f"Soru: {entry['question']}")
            print(f"Cevap: {entry['answer']}")
            print("-" * 50)
    
    def show_suggestions(self) -> None:
        """Önerilen soruları göster"""
        print("\nÖnerilen Sorular:")
        for category, questions in self.suggestions.items():
            print(f"\n{category}:")
            print("-" * len(category))
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")
    
    def add_suggestion(self, category: str, suggestion: str) -> None:
        """Yeni soru önerisi ekle"""
        if category not in self.suggestions:
            self.suggestions[category] = []
        
        if suggestion not in self.suggestions[category]:
            self.suggestions[category].append(suggestion)
            print(f"\nYeni soru önerisi eklendi: {suggestion} ({category} kategorisinde)")

    def generate_quiz(self):
        """Quiz oluştur ve göster"""
        try:
            # Tüm oturumlardaki başarılı cevapları al
            all_histories = self.load_all_histories()
            
            if not all_histories:
                print("\nHenüz yeterli soru-cevap geçmişi yok! Önce birkaç soru sorun.")
                return
                
            response = requests.post(f"{self.url}/generate-quiz", json={"history_file": self.history_file})
            if response.status_code == 200:
                quiz = response.json()
                print(f"\n{quiz['title']}")
                print("=" * 50)
                
                for i, q in enumerate(quiz['questions'], 1):
                    print(f"\nSoru {i}: {q['question']}")
                    for option in q['options']:
                        print(option)
                    print(f"\nDoğru Cevap: {q['correct_answer']}")
                    print("-" * 50)
            elif response.status_code == 404:
                print("\nHenüz yeterli soru-cevap geçmişi yok! Önce birkaç soru sorun.")
            else:
                print(f"\nQuiz oluşturulurken bir hata oluştu: {response.json()['detail']}")
        except Exception as e:
            print(f"\nHata: {str(e)}")

async def main():
    client = RAGClient()
    
    # Otomatik klasör yükle (buraya kendi path'inizi yazın)
    auto_dir_path = "/Users/yaseminarslan/Desktop/mcp/dokumanlar"
    if os.path.isdir(auto_dir_path):
        result = client.load_files(auto_dir_path)
        print(f"\n{result['message']}")
    else:
        print(f"\nOtomatik yükleme için klasör bulunamadı: {auto_dir_path}")
    
    print("Teknik RAG Sistemi Soru-Cevap Uygulaması")
    print("Komutlar:")
    print("q veya question: Soru sor")
    print("h veya history: Geçmiş soruları göster")
    print("s veya suggestions: Önerilen soruları göster")
    print("a veya add: Yeni soru ekle")
    print("u veya upload: Dosya yükle")
    print("l veya load: Klasörden dosya yükle")
    print("z veya quiz: Quiz oluştur")
    print("exit: Çıkış")
    print("-" * 50)
    
    while True:
        try:
            command = input("\nKomut girin: ").strip().lower()
            
            if command in ['q', 'question']:
                question = input("Sorunuzu girin: ")
                # Soruyu gönder ve cevabı al
                answer = await client.ask_question(question)
                print("\nCevap:", answer)
                
            elif command in ['h', 'history']:
                client.show_history()
            elif command in ['s', 'suggestions']:
                client.show_suggestions()
            elif command in ['a', 'add']:
                print("\nMevcut kategoriler:")
                for i, category in enumerate(client.suggestions.keys(), 1):
                    print(f"{i}. {category}")
                print("Yeni kategori için 'new' yazın")
                
                category = input("\nKategori seçin veya yeni kategori adı girin: ").strip()
                if category.lower() == 'new':
                    category = input("Yeni kategori adı: ").strip()
                
                new_suggestion = input("Yeni soru önerisi: ").strip()
                if new_suggestion:
                    client.add_suggestion(category, new_suggestion)
            elif command in ['u', 'upload']:
                file_path = input("Dosya yolunu girin: ")
                if os.path.exists(file_path):
                    result = client.upload_file(file_path)
                    print(f"\n{result['message']}")
                else:
                    print("\nDosya bulunamadı!")
            elif command in ['l', 'load']:
                directory = input("Klasör yolunu girin: ")
                if os.path.isdir(directory):
                    result = client.load_files(directory)
                    print(f"\n{result['message']}")
                else:
                    print("\nKlasör bulunamadı!")
            elif command in ['z', 'quiz']:
                client.generate_quiz()
            elif command == 'exit':
                print("\nOturum sonlandırılıyor...")
                if client.history:  # Eğer geçmişte soru varsa
                    print("\nQuiz oluşturmak ister misiniz? (E/H)")
                    if input().lower() == 'e':
                        client.generate_quiz()
                else:
                    print("\nGeçmişte soru bulunmadığı için quiz oluşturulamaz.")
                print("\nProgram sonlandırılıyor...")
                break
            else:
                print("Geçersiz komut!")
            
        except requests.exceptions.ConnectionError:
            print("\nHata: RAG sunucusuna bağlanılamadı!")
            print("Lütfen rag_server.py'nin çalışır durumda olduğundan emin olun.")
            break
            
        except Exception as e:
            print(f"\nBir hata oluştu: {str(e)}")
            continue

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 