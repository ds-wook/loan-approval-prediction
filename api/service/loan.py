import time
from pathlib import Path
from typing import List

import lightgbm as lgb
import pandas as pd

from api.dto.loan import LoanApprovalDto


async def run_model(data: LoanApprovalDto) -> List[float]:
    data_dic = data.dict()

    # dictionary 형태를 DataFrame 형태로 변환한다.
    input_data = pd.DataFrame.from_dict([data_dic], orient="columns")

    # 학습시킨 모델의 가중치 파일을 불러와서 모델에 적용시킨다.
    predicted_start = time.time()
    loaded_model = lgb.Booster(model_file=Path("res/models/") / "lightgbm_model.txt")
    lgb_preds = loaded_model.predict(input_data)
    predicted_elapsed = 1000 * (time.time() - predicted_start)

    # 변수 타입이 numpy이기 때문에 list로 바꿔준다.
    result = lgb_preds.tolist()

    response = {"prediction": result, "prediction_elapsed": predicted_elapsed}

    return response
