# OpenAI Vector Store Toolkit CLI

Một giao diện dòng lệnh (CLI) để quản lý và tương tác với các Kho Vector (Vector Stores) và tệp của OpenAI. Công cụ này được xây dựng dựa trên các dịch vụ đã có trong codebase của bạn, giúp bạn thao tác nhanh chóng với các chức năng RAG.

## Yêu cầu

- Python 3.12+
- Một API Key từ OpenAI

## Cài đặt

1.  **Clone repository về máy của bạn:**
    ```bash
    git clone <URL_CUA_REPOSITORY_CUA_BAN>
    cd openai_vstore_kit
    ```

2.  **Tạo và kích hoạt môi trường ảo (Virtual Environment):**
    Rất khuyến khích bạn nên cài đặt các gói phụ thuộc trong một môi trường ảo để tránh xung đột với các thư viện hệ thống.
    ```bash
    # Tạo môi trường ảo
    python -m venv .venv
    # Kích hoạt môi trường ảo
    # Trên Windows
    .\.venv\Scripts\activate
    # Trên macOS/Linux
    source .venv/bin/activate
    ```

3.  **Cài đặt các gói phụ thuộc:**
    Dự án này sử dụng `hatchling`. Bạn có thể dùng `pip` hoặc `uv` để cài đặt.

    -   **Sử dụng `pip`:**
        ```bash
        # Cài đặt thông thường
        pip install .
        # Cài đặt ở chế độ có thể chỉnh sửa (editable)
        pip install -e .
        ```

    -   **Sử dụng `uv` (trình cài đặt tốc độ cao):**
        ```bash
        # Cài đặt thông thường
        uv pip install .
        # Cài đặt ở chế độ có thể chỉnh sửa (editable)
        uv pip install -e . hoặc uv sync
        ```

## Cấu hình

Công cụ này yêu cầu một API key của OpenAI để hoạt động.

1.  **Tạo file `.env`:**
    Sao chép file `.env.example` thành một file mới có tên là `.env`.
    ```bash
    cp .env.example .env
    ```
    ```powershell
    copy-item .env.example .env
    ```

2.  **Thêm API Key:**
    Mở file `.env` và thay thế `<your-api-key>` bằng API Key của bạn.
    ```ini
    OPENAI_API_KEY=<your-api-key>
    # OPENAI_BASE_URL=<your_base_url> # Tùy chọn
    ```

## Giới thiệu các Services

Kiến trúc của toolkit này được chia thành các service module, mỗi service đảm nhận một nhiệm vụ cụ thể:

-   **`StoreService`**: Quản lý vòng đời của các Vector Stores. Bao gồm các chức năng tạo, xóa, liệt kê và tìm kiếm kho vector theo tên.
-   **`FileService`**: Quản lý tệp trong một Vector Store cụ thể. Hỗ trợ tải tệp lên từ local hoặc URL, thêm/xóa tệp khỏi kho, cập nhật metadata, và thực hiện tìm kiếm ngữ nghĩa (semantic search).
-   **`ConversationService`**: Quản lý các cuộc hội thoại (conversations) với OpenAI API, cho phép tạo, truy xuất, cập nhật và xóa các phiên hội thoại.
-   **`ResponseRAGService`**: Một lớp bao bọc (wrapper) chuyên dụng cho việc tạo ra các câu trả lời RAG (Retrieval-Augmented Generation) bằng cách sử dụng `file_search` tool trong một cuộc hội thoại cụ thể.

## Hướng dẫn sử dụng chi tiết

Lệnh chính của công cụ là `vstore`. Nó được chia thành hai nhóm lệnh con: `store` và `file`.

Bạn có thể xem trợ giúp chi tiết cho bất kỳ lệnh nào bằng cách thêm cờ `--help`. Điều này rất hữu ích để xem tất cả các tùy chọn có sẵn.
**Ví dụ:** `vstore --help`, `vstore store --help`, `vstore file upload-and-add --help`.

---


### Quản lý Kho Vector (`store`)

Các lệnh để tạo, liệt kê và xóa kho vector.

-   **`vstore store get-or-create <TEN_KHO>`**
    Lấy `store_id` nếu kho đã tồn tại, nếu không sẽ tạo một kho mới với tên được cung cấp.
    ```bash
    vstore store get-or-create "My-Awesome-Store"
    ```

-   **`vstore store list`**
    Liệt kê tất cả các kho vector có trong tài khoản của bạn.
    ```bash
    vstore store list
    ```

-   **`vstore store get-id-by-name <TEN_KHO>`**
    Tìm và trả về `store_id` dựa vào tên của kho.
    ```bash
    vstore store get-id-by-name "My-Awesome-Store"
    ```

-   **`vstore store delete <STORE_ID>`**
    Xóa một kho vector dựa vào ID của nó.
    ```bash
    vstore store delete "vs_abc123"
    ```

---


### Quản lý Tệp (`file`)

Các lệnh để tải lên, liệt kê, xóa và truy vấn tệp trong một kho vector cụ thể.

-   **`vstore file list <STORE_ID>`**
    Liệt kê tất cả các tệp trong một kho vector.
    ```bash
    vstore file list "vs_abc123"
    ```

-   **`vstore file upload-and-add <STORE_ID> <DUONG_DAN_FILE_HOAC_URL>`**
    Tải một tệp từ máy tính hoặc một URL và thêm nó vào kho vector.
    -   `--attr <key=value>`: Gắn các thuộc tính (metadata) cho tệp (có thể dùng nhiều lần).
    -   `--max-chunk-size <int>`: Kích thước chunk tối đa (mặc định: 800).
    -   `--chunk-overlap <int>`: Độ chồng chéo giữa các chunk (mặc định: 400).

    *Ví dụ với tệp cục bộ:*
    ```bash
    vstore file upload-and-add "vs_abc123" "./docs/handbook.pdf" --attr source=internal --attr lang=vi
    ```

-   **`vstore file find-id-by-name <STORE_ID> <TEN_FILE>`**
    Tìm `vector_store_file_id` của một tệp dựa vào tên gốc của nó.
    ```bash
    vstore file find-id-by-name "vs_abc123" "handbook.pdf"
    ```

-   **`vstore file update-attrs <STORE_ID> <VECTOR_STORE_FILE_ID>`**
    Cập nhật thuộc tính cho một tệp đã có trong kho.
    ```bash
    vstore file update-attrs "vs_abc123" "vsf_xyz456" --attr owner=data-team
    ```

-   **`vstore file delete <STORE_ID> <VECTOR_STORE_FILE_ID>`**
    Xóa một tệp khỏi kho vector.
    ```bash
    vstore file delete "vs_abc123" "vsf_xyz456"
    ```

-   **`vstore file semantic-retrieve <STORE_ID> "<CAU_TRUY_VAN>"`**
    Thực hiện tìm kiếm ngữ nghĩa (semantic search) trên kho vector với một câu truy vấn.
    ```bash
    vstore file semantic-retrieve "vs_abc123" "Chính sách nghỉ phép năm 2024 là gì?"
    ```

## Mini Report: Đánh giá năng lực RAG dựa trên tài liệu "Deep Research Blog"

Một bài kiểm tra gồm 10 câu hỏi đã được thực hiện để đánh giá khả năng của mô hình RAG trong việc trả lời các câu hỏi dựa trên tài liệu *Deep Research Blog*. Kết quả được đánh giá theo bộ tiêu chí 6 điểm: `Chính xác`, `Đầy đủ`, `Rõ ràng`, `Tuân thủ nguồn`, `Truy vết`, và `Trình bày`.

-   **Kết quả tổng quan:** Mô hình đạt hiệu suất xuất sắc với tổng điểm trung bình là **9.8/10**.
    -   **8/10** câu hỏi đạt điểm tuyệt đối **10/10**.
    -   **2/10** câu hỏi đạt điểm **9/10**.

-   **Phân tích điểm mạnh:** Mô hình thể hiện năng lực vượt trội ở hầu hết các tiêu chí, đặc biệt là:
    -   **Chính xác & Tuân thủ nguồn:** Tất cả các câu trả lời đều chính xác và hoàn toàn bám sát vào nội dung của tài liệu nguồn.
    -   **Rõ ràng & Truy vết:** Câu trả lời được trình bày mạch lạc, dễ hiểu và có khả năng truy vết lại nguồn thông tin.

-   **Điểm cần cải thiện:**
    -   **Tính đầy đủ (Completeness):** Hai câu trả lời bị trừ điểm ở tiêu chí này do thiếu một vài chi tiết phụ, dù không ảnh hưởng đến tính chính xác của nội dung cốt lõi.
        -   **Câu hỏi 3:** Bỏ lỡ chi tiết "tương đương một nhà phân tích" khi mô tả về khả năng của Deep Research.
        -   **Câu hỏi 9:** Không nhấn mạnh đủ về hiệu suất của mô hình trên "các benchmark khó" như trong tài liệu gốc.

-   **Kết luận:** Mô hình RAG hoạt động rất hiệu quả và đáng tin cậy. Điểm cần cải thiện duy nhất là khả năng bao quát toàn bộ các chi tiết nhỏ trong tài liệu để đạt được sự hoàn hảo về tính đầy đủ.
