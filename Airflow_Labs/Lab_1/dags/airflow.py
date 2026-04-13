from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, evaluate_model

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Iris_Classification_Lab1',
    default_args=default_args,
    description='Iris classification pipeline for Lab 1 of Airflow series',
    catchup=False,
) as dag:

    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.pkl"],
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model,
        op_args=["model.pkl", build_save_model_task.output],
    )

    load_data_task >> data_preprocessing_task >> build_save_model_task >> evaluate_model_task

if __name__ == "__main__":
    dag.test()
