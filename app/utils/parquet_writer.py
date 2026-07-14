import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Iterator, List

class ParquetWriterUtil:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.schema = pa.schema([
            ("text",         pa.string()),
            ("source",       pa.string()),
            ("token_length", pa.int64()),
            ("meta",         pa.string()),
        ])
        self.writer = pq.ParquetWriter(self.output_path, self.schema, compression='snappy')

    def run(self, generator: Iterator[List[Dict]]):
        """Pipeline bây giờ chỉ việc nhận batch từ generator và ghi xuống."""
        print(f"Bắt đầu ghi dữ liệu vào: {self.output_path}")
        total = 0

        for batch in generator:
            table = pa.Table.from_pylist(batch, schema=self.schema)
            self.writer.write_table(table)
            total += len(batch)
            print(f"Đã ghi batch {len(batch)} dòng. Tổng cộng: {total}")

        print(f"Thành công! Tổng file có {total} dòng.")

    def close(self):
        if self.writer:
            self.writer.close()
                
if __name__ == "__main__":
    from app.trading.gen.rawchart_gen import RawChartGenerator
    
    input_files = [
        "data/encoded/AUDUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/AUDUSD_H1_preprocessed_encoded.csv",
        "data/encoded/EURUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/EURUSD_H1_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/GBPUSD_H1_preprocessed_encoded.csv",
        "data/encoded/US500_1Min_preprocessed_encoded.csv",
        "data/encoded/US500_5Min_preprocessed_encoded.csv",
        "data/encoded/US500_15Min_preprocessed_encoded.csv",
        "data/encoded/US500_H1_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_1Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_5Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_15Min_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_H1_preprocessed_encoded.csv",
        "data/encoded/XAUUSD_Daily_preprocessed_encoded.csv",
    ]
    
    pawriter = ParquetWriterUtil("data/dataset/train_raw_100candles.parquet")
    for f in input_files:
        print(f"Processing {f}")
        pawriter.run(RawChartGenerator(f))