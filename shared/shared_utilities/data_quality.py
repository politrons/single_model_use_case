from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def get_all_flag_names(primary_keys: list[str]) -> dict[str, str]:

    all_flags = {
        c: f'{c}_null_dq_flag'
        for c in primary_keys
    }

    all_flags['duplicated_records'] = 'duplicated_row_dq_flag'

    return all_flags

def add_valid_row_and_dq_flags(
    spark_or_df: SparkSession | DataFrame,
    table_name: str,
    primary_keys: list[str],
) -> DataFrame:
    
    all_flags = get_all_flag_names(primary_keys)
    null_flags = [
        F.col(c).isNull().cast('boolean').alias(all_flags[c])
        for c in primary_keys
    ]
    any_problem = F.lit(False)
    for c in all_flags.values():
        any_problem = any_problem | F.col(c)

    if isinstance(spark_or_df, SparkSession):
        df = spark_or_df.table(table_name)
    elif isinstance(spark_or_df, DataFrame):
        df = spark_or_df
    else:
        raise TypeError('Unrecognised type')

    df = (
        df
        .select(
            '*',
            *null_flags
        )
        .withColumn(
            all_flags['duplicated_records'],
            (F.row_number().over(Window.partitionBy(primary_keys).orderBy(primary_keys))) > 1
        )
        .withColumn(
            'valid_row',
            ~any_problem
        )
    )

    return df

def get_dq_count(
    spark: SparkSession,
    table_name: str,
    primary_keys: list[str],
) -> DataFrame:
    return (
        spark
        .table(table_name)
        .where(~F.col('valid_row'))
        .select([
            F.coalesce(F.sum(F.col(c).cast('int')), F.lit(0)).alias(c)
            for c in get_all_flag_names(primary_keys).values()
        ])
    )

def get_dq_rules(primary_keys: list[str]) -> dict[str, str]:

    all_checks = {
        f'{dq_name}_null_records' if dq_name in primary_keys else dq_name: f'{dq_flag} = FALSE'
        for dq_name, dq_flag in get_all_flag_names(primary_keys).items()
    }

    return all_checks
