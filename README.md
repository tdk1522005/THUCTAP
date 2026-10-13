# THUCTAP — Real-Time Speech Translation

Ứng dụng nhận diện giọng nói tiếng Anh, dịch sang tiếng Việt và tổng hợp giọng nói tiếng Việt. Pipeline web có thể dùng Gemini (mặc định) hoặc OpenAI; các module local và benchmark hiện có được giữ nguyên. Bản web hiện xử lý theo cụm audio khi phát hiện khoảng lặng, chưa phải luồng transcript từng token.

## 1. Công nghệ sử dụng

- **Python 3.12**
- **Speech-to-Text (STT):** Gemini API, OpenAI API và Faster-Whisper
- **Machine Translation:** Gemini API, OpenAI API và mô hình dịch local
- **Text-to-Speech (TTS):** Gemini API và OpenAI API
- **Web:** Flask, Flask-SocketIO
- **Audio processing:** SoundFile, NumPy, imageio-ffmpeg
- **Testing:** pytest

## 2. Yêu cầu hệ thống

- Python 3.12 hoặc phiên bản tương thích với các dependencies.
- Git.
- Kết nối Internet và API key của nhà cung cấp đang chọn (Gemini mặc định; OpenAI là phương án dự phòng).
- Model đã cấu hình phải được tài khoản API của bạn hỗ trợ.
- Microphone và loa/tai nghe nếu sử dụng chức năng thu âm và phát giọng nói.

Các script sử dụng mô hình local có thể cần tải model lần đầu và tiêu tốn RAM/CPU đáng kể.

## 3. Cài đặt project

### Bước 1: Clone repository

```powershell
git clone https://github.com/tdk1522005/THUCTAP.git
cd THUCTAP
```

### Bước 2: Tạo môi trường ảo

```powershell
python -m venv .venv
```

Kích hoạt môi trường ảo trong PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn việc chạy script, có thể sử dụng Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Bước 3: Cài đặt dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-gemini.txt
```

Nếu cài đặt gặp lỗi, hãy kiểm tra phiên bản Python và thông báo lỗi trước khi cài thêm package thủ công.

## 4. Cấu hình Gemini (mặc định) hoặc OpenAI

Tạo file `.env` từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Mở `.env`. Để dùng Gemini, cấu hình:

```env
AI_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_STT_MODEL=gemini-2.5-flash
GEMINI_TRANSLATION_MODEL=gemini-2.5-flash
GEMINI_TTS_MODEL=gemini-3.8-flash-tts
GEMINI_TTS_VOICE=Kore
```

Muốn chuyển về OpenAI, đặt `AI_PROVIDER=openai` và điền `OPENAI_API_KEY` cùng các biến model OpenAI tương ứng trong `.env.example`. Không chia sẻ API key hoặc commit file `.env` lên GitHub.

**Lưu ý:** model có thể phụ thuộc quyền truy cập/tình trạng cung cấp của tài khoản. Chưa thực hiện API call thật trong bước tích hợp này; unit test dùng mock và không tiêu hao API credit. Các API có thể phát sinh chi phí khi chạy ứng dụng.

## 5. Chạy giao diện web

Sau khi hoàn thành cài đặt và cấu hình:

```powershell
python app.py
```

Mở trình duyệt tại:

http://127.0.0.1:5000

Endpoint kiểm tra trạng thái:

http://127.0.0.1:5000/api/health

Giao diện web sử dụng các module STT, dịch thuật và TTS được cấu hình trong project. Một số chức năng cần kết nối Internet và API key hợp lệ.

Nếu giao diện không kết nối được với backend, hãy kiểm tra cửa sổ terminal để xem lỗi Flask hoặc Flask-SocketIO.

## 6. Chạy kiểm thử

Để chạy các unit test:

```powershell
python -m pytest tests/ -q
```

Để xem chi tiết từng test:

```powershell
python -m pytest tests/ -v
```

Các test sử dụng mock để kiểm tra một số module OpenAI mà không cần gọi API thật. Tuy nhiên, không nên mặc định mọi test hoặc script trong repository đều miễn phí: hãy kiểm tra nội dung trước khi chạy.

## 7. Cấu trúc thư mục

```text
THUCTAP/
├── app.py
├── record.py
├── requirements.txt
├── .env.example
├── .gitignore
├── evaluation_audio/
├── evaluation_results/
├── stt/
│   ├── benchmark_openai_stt.py
│   ├── benchmark_stt.py
│   ├── evaluate_stt.py
│   ├── openai_transcriber.py
│   ├── realtime_stt.py
│   ├── sentence_buffer.py
│   ├── transcriber.py
│   └── transcript_stabilizer.py
├── translation/
│   ├── benchmark_openai_translation.py
│   ├── benchmark_translation.py
│   ├── evaluate_translation_bleu.py
│   ├── openai_translator.py
│   ├── openai_tts.py
│   └── translator.py
├── templates/
│   └── index.html
└── tests/
    ├── test_evaluate_stt.py
    ├── test_openai_transcriber.py
    ├── test_openai_translator.py
    ├── test_openai_tts.py
    ├── test_realtime_stt.py
    └── test_translator.py
```

Danh sách trên thể hiện các file tiêu biểu; repository còn có những file test, benchmark và dữ liệu phục vụ đánh giá khác.

### Mô tả các thành phần

| Thành phần | Chức năng |
|---|---|
| `app.py` | Backend Flask và các endpoint/giao tiếp realtime |
| `templates/` | Giao diện web |
| `stt/` | Nhận diện giọng nói, xử lý transcript và đánh giá STT |
| `translation/` | Dịch thuật, tổng hợp giọng nói và đánh giá dịch |
| `evaluation_audio/` | Các mẫu âm thanh dùng trong thực nghiệm |
| `evaluation_results/` | Kết quả benchmark và đánh giá |
| `tests/` | Unit test và kiểm thử tích hợp |
| `requirements.txt` | Danh sách dependencies |
| `.env.example` | Mẫu cấu hình biến môi trường |

## 8. Chạy benchmark và đánh giá

Project có các script đánh giá STT và dịch thuật trong thư mục `stt/` và `translation/`.

Ví dụ:

```powershell
python -m stt.evaluate_stt
python -m translation.evaluate_translation_bleu
```

Hai lệnh trên chỉ là ví dụ về cách chạy module. Hãy kiểm tra phần `if __name__ == "__main__"` và cách sử dụng của từng script trước khi chạy; không phải mọi module đều được thiết kế để chạy trực tiếp theo cách này.

Các script benchmark OpenAI có thể gửi yêu cầu API thật và phát sinh chi phí. Các script dùng Faster-Whisper hoặc mô hình dịch local có thể tải model về máy.

## 9. Xử lý sự cố thường gặp

### Không tìm thấy API key

- Kiểm tra file `.env` nằm ở thư mục gốc.
- Kiểm tra tên biến `OPENAI_API_KEY`.
- Không thêm dấu cách hoặc ký tự thừa vào API key.

### Không import được package

Kích hoạt đúng môi trường ảo và cài lại dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Không truy cập được giao diện web

- Kiểm tra terminal có báo lỗi khi chạy `python app.py` hay không.
- Kiểm tra địa chỉ `http://127.0.0.1:5000`.
- Kiểm tra cổng 5000 có đang được chương trình khác sử dụng không.
- Kiểm tra phiên bản Flask-SocketIO và các thư viện liên quan.

### Nhận diện hoặc dịch quá chậm

Độ trễ có thể đến từ việc gửi âm thanh lên API, xử lý mô hình, dịch văn bản, tổng hợp giọng nói hoặc thời gian chờ giữa các đoạn. Nên đo riêng từng công đoạn trước khi thay đổi cấu hình.

## 10. Bảo mật

- Không commit `.env`, API key hoặc thông tin xác thực.
- Không công khai bản ghi âm chứa dữ liệu cá nhân khi chưa được phép.
- Không chạy các script benchmark tốn phí khi chưa kiểm tra cách hoạt động.
- Khi triển khai ngoài máy cá nhân, cần cấu hình xác thực, HTTPS và giới hạn truy cập phù hợp.

## 11. Trạng thái kiểm thử

Tại thời điểm cập nhật tài liệu, bộ test đã chạy thành công với kết quả:

```text
28 passed
```

Thời gian chạy ghi nhận ở một lần kiểm thử là 5.90 giây. Đây là kết quả tại thời điểm kiểm tra, không phải cam kết về thời gian chạy trên mọi máy tính.

---

**Repository:** https://github.com/tdk1522005/THUCTAP
