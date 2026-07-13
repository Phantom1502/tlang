from app.lang.execution import Interpreter
from app.lang.ast import Parser
from app.lang.lexical import Lexer

# =====================================================================
# CHƯƠNG TRÌNH CHẠY THỬ (DEMO)
# =====================================================================
if __name__ == "__main__":
    # Biểu thức kiểm tra (hỗ trợ cả độ ưu tiên toán tử và dấu ngoặc)
    code = "<chart> + <chart> + <chart> + <chart> + <chart> + <chart>"
    
    print(f"Mã nguồn: {code}\n")

    # 1. Chạy Lexer
    lexer = Lexer(code)
    
    # 2. Chạy Parser để dựng AST
    parser = Parser(lexer)
    ast_tree = parser.parse()
    print(f"Cây cấu trúc (AST): {ast_tree}\n")