from pydantic import BaseModel


class LoanApprovalDto(BaseModel):
    person_home_ownership: int | float
    loan_intent: int | float
    loan_grade: int | float
    cb_person_default_on_file: int | float
    person_age: int | float
    person_income: int | float
    person_emp_length: int | float
    loan_amnt: int | float
    loan_int_rate: int | float
    loan_percent_income: int | float
    cb_person_cred_hist_length: int | float
