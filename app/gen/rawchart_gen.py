
import json
import pandas as pd
from app.gen.base_gen import BaseGenerator
from app.preprocess.candle import CandleParser
from typing import List, Dict
import random

class RawChartGenerator(BaseGenerator):
    def __init__(self, csv_data: str, tokenizer=None):
        super().__init__(tokenizer)
        self.df = pd.read_csv(csv_data)
        self.num_samples = len(self.df)
        self._count = 0

    def __next__(self) -> List[Dict]:
        if self._count >= self.num_samples:
            raise StopIteration

        batch = []
        while len(batch) < self.batch_size:
            if self._count >= self.num_samples:
                break
            
            gen_text = self.df.iloc[self._count]["text"]
            candleParser = CandleParser(gen_text)
            token_length = len(self.tokenizer.encode(gen_text)) if self.tokenizer else 402

            meta = {
                "type": "raw_chart",
                "anchor_open": self.df.iloc[self._count]["anchor_open"],
                "anchor_atr": self.df.iloc[self._count]["anchor_atr"]
            }

            batch.append({
                "text": candleParser.build_raw_text(),
                "source": "trading",
                "token_length": token_length,
                "meta": json.dumps(meta, ensure_ascii=False)
            })
            self._count += 1
            
            min_value = candleParser.min()
            max_value = candleParser.max()

            low_distance = min_value
            high_distance = 1023 - max_value
            ranges = [-low_distance, high_distance]
            # randomly 10 times select a range excluding 0
            valid_numbers = [x for x in range(-low_distance, high_distance + 1) if x != 0]

            K = min(10, len(valid_numbers)) 

            result = random.sample(valid_numbers, K)
            for r in result:
                rparser = CandleParser(gen_text)
                rparser.shift(r)
                rtext = rparser.build_raw_text()
                batch.append({
                    "text": rtext,
                    "source": "trading",
                    "token_length": token_length,
                    "meta": json.dumps(meta, ensure_ascii=False)
                })
                
            
        return batch