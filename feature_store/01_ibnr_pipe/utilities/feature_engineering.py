import sys, os

_cwd = os.getcwd()
_shared = os.path.normpath(os.path.join(_cwd, "../../../shared"))
sys.path.insert(0, _shared)

from ibnr_utilities.utils import get_occured_date_col
from pyspark.sql import functions as F
from pyspark.sql import DataFrame, Window

LOWEST_GRANULARITY_PROPERTIES = {
    "PolicySubType": "contract",
    "AgeBucket": "contract",
    "UKMemberRegion": "contract",
    "SystemID": "invoice",
    "MajorICDGroupingDescription": "invoice",
    # "CareMarkerDescription": "invoice",  #### REMOVED AS REQUESTED BY CAMILLE
}


def make_chain_ladder_format(
    df_invoice: DataFrame,
    occured_date_col: str,
    lag_col: str,
    seg_cols: list[str],
    maximal_years_runoff_lags: int,
) -> DataFrame:
    seg_cols_list = list(seg_cols)

    lowest_granularity = [occured_date_col, *LOWEST_GRANULARITY_PROPERTIES.keys()]

    # Step 1: Get all unique combinations to build target grain
    df_invoice = df_invoice.where((F.col(lag_col) <= maximal_years_runoff_lags * 12) & (F.col(lag_col) >= 0)).select(
        *lowest_granularity,
        lag_col,
        "amount_paid",
        "claim_id",
        "invoice_id",
    )

    df_all_timeline = df_invoice.select(*lowest_granularity).distinct()
    lag_values = df_invoice.select(lag_col).distinct()
    df_all_timeline = df_all_timeline.crossJoin(F.broadcast(lag_values))

    # Step 2: Self-join to create all (Date1, SystemID, TargetLag, DataLag) with DataLag <= TargetLag
    date_lag_pairs = (
        df_all_timeline.alias("a")
        .join(
            df_all_timeline.alias("b"),
            [F.col(f"a.{c}") == F.col(f"b.{c}") for c in lowest_granularity] + [F.col(f"b.{lag_col}") <= F.col(f"a.{lag_col}")],
        )
        .select(*[F.col(f"a.{c}").alias(c) for c in lowest_granularity], F.col(f"a.{lag_col}").alias(lag_col), F.col(f"b.{lag_col}").alias("DataLag"))
    )

    # Step 3: Join with the main table to bring in amount_paid and claim_id
    joined = (
        date_lag_pairs.alias("a")
        .join(
            df_invoice.alias("b"),
            [F.col(f"a.{c}") == F.col(f"b.{c}") for c in lowest_granularity] + [F.col(f"b.{lag_col}") == F.col("a.DataLag")],
            how="left",
        )
        .select(
            *[F.col(f"a.{c}") for c in lowest_granularity],
            F.col(f"a.{lag_col}"),
            F.col("b.amount_paid"),
            F.col("b.claim_id"),
            F.col("b.invoice_id"),
        )
    )

    # Step 4: Aggregate
    df_chain_ladder = joined.groupBy(*lowest_granularity, lag_col).agg(
        F.sum("amount_paid").alias("cumulative_amount"),
        F.countDistinct("claim_id").alias("cumulative_nb_claim"),
        F.countDistinct("invoice_id").alias("cumulative_nb_invoice"),
    )

    df_chain_ladder = (
        df_chain_ladder.groupBy(occured_date_col, lag_col, *seg_cols_list)
        .agg(
            F.sum("cumulative_amount").alias("cumulative_amount"),
            F.sum("cumulative_nb_claim").alias("cumulative_nb_claim"),
            F.sum("cumulative_nb_invoice").alias("cumulative_nb_invoice"),
        )
        .orderBy(occured_date_col, lag_col, *seg_cols_list)
    )

    df_chain_ladder = df_chain_ladder.fillna(0)

    # Window function instead of separate join for max values
    window_spec = Window.partitionBy(occured_date_col, *seg_cols_list).orderBy(F.col(lag_col).desc())

    df_chain_ladder = (
        df_chain_ladder.withColumn("ultimate_amount", F.first("cumulative_amount").over(window_spec))
        .withColumn("ultimate_nb_invoice", F.first("cumulative_nb_invoice").over(window_spec))
        .withColumn("ultimate_nb_claim", F.first("cumulative_nb_claim").over(window_spec))
        .orderBy(occured_date_col, lag_col, *seg_cols_list)
    )

    return df_chain_ladder


def maximal_feature_engineering(
    df_invoice: DataFrame,
    df_preauth: DataFrame,
    segmentation_features: list[str],
    max_lags: int = 12,
    maximal_years_runoff_lags: int = 3,
) -> DataFrame:
    segmentation_features_list = list(segmentation_features)
    occured_date_col = get_occured_date_col()

    # Aggregating invoices into Cumulative Paid format
    df_invoice_agg_CP = make_chain_ladder_format(
        df_invoice.where(F.col("payment_date").isNotNull()),
        occured_date_col,
        "lag_treatment_payment",
        segmentation_features,
        maximal_years_runoff_lags,
    )
    df_invoice_agg_CP = df_invoice_agg_CP.withColumnRenamed("lag_treatment_payment", "lag")

    df_invoice_agg_CP = df_invoice_agg_CP.select(*[F.col(c).alias(f"{c}_paid") if "cumulative" in c or "ultimate" in c else F.col(c) for c in df_invoice_agg_CP.columns])

    df_invoice_agg_CP = df_invoice_agg_CP.select(*segmentation_features_list, occured_date_col, "lag", *[c for c in df_invoice_agg_CP.columns if "cumulative" in c or "ultimate" in c])

    # Aggregating invoices into Cumulative Received format
    df_invoice_agg_CR = make_chain_ladder_format(
        df_invoice,
        occured_date_col,
        "lag_treatment_reception",
        segmentation_features,
        maximal_years_runoff_lags,
    )
    df_invoice_agg_CR = df_invoice_agg_CR.withColumnRenamed("lag_treatment_reception", "lag")

    df_invoice_agg_CR = df_invoice_agg_CR.select(*[F.col(c).alias(f"{c}_received") if "cumulative" in c else F.col(c) for c in df_invoice_agg_CR.columns])

    df_invoice_agg_CR = df_invoice_agg_CR.select(*segmentation_features_list, occured_date_col, "lag", *[c for c in df_invoice_agg_CR.columns if "cumulative" in c])

    # Joining
    df_invoice_agg = df_invoice_agg_CP.join(df_invoice_agg_CR, on=segmentation_features_list + [occured_date_col, "lag"], how="full_outer")
    for c in df_invoice_agg.columns:
        if "cumulative" in c:
            df_invoice_agg = df_invoice_agg.fillna({c: 0})

    # Extra feature eng
    select_expr = [(F.col(c) - F.col(c.replace("received", "paid"))).alias(f"{c}_but_not_paid") if "cumulative" in c and "received" in c and "nb" in c else F.col(c) for c in df_invoice_agg.columns]
    df_invoice_agg = df_invoice_agg.select(*select_expr)

    # Getting Avg processing times
    processing_seg_level = []  # processing time should only be influenced by system
    df_invoice = df_invoice.where(F.col("daily_lag_reception_payment") >= 0).where(F.col("daily_lag_reception_payment") <= 93)

    df_processing_time = df_invoice.groupBy("payment_date", *processing_seg_level).agg(F.mean("daily_lag_reception_payment").alias("avg_monthly_processing_time"))
    df_invoice_agg = (
        df_invoice_agg.withColumn("payment_date", F.add_months(occured_date_col, F.col("lag")))
        .join(df_processing_time, on=["payment_date"] + processing_seg_level, how="left")
        .drop("payment_date")
        .fillna({"avg_monthly_processing_time": 0.0})
    )

    ######################################## modified by gustavo.martins
    pivot_mappings = [1, 2, 3, 4]
    pivot_col = "week_id"
    value_col = "avg_weekly_processing_time"
    pivot_expr = [F.first(F.when(F.col(pivot_col) == val, F.col(value_col)), ignorenulls=True).alias(f"week_{val}") for val in pivot_mappings]
    ######################################## end

    df_processing_time = (
        df_invoice.withColumn("nb_removable_days", F.day(F.last_day("payment_date")) % 7)
        .where(F.col("payment_date_full") >= F.date_add("payment_date", F.col("nb_removable_days")))
        .withColumn("week_id", (F.floor((F.day("payment_date_full") - F.col("nb_removable_days") - 1) / 7) + 1))
        ######################################## modified by gustavo.martins
        .withColumn("week_id", F.col("week_id").cast("int"))
        ######################################## end
        .groupBy("payment_date", *processing_seg_level, "week_id")
        .agg(F.mean("daily_lag_reception_payment").alias("avg_weekly_processing_time"))
        .groupBy("payment_date", *processing_seg_level)
        ######################################## modified by gustavo.martins
        # .pivot("week_id")
        # .agg({"AvgWeeklyProcessingTime": "first"})
        # .withColumnRenamed("1", "AvgProcessingTimeW1")
        # .withColumnRenamed("2", "AvgProcessingTimeW2")
        # .withColumnRenamed("3", "AvgProcessingTimeW3")
        # .withColumnRenamed("4", "AvgProcessingTimeW4")
        .agg(*pivot_expr)
        .withColumnRenamed("week_1", "avg_processing_time_w1")
        .withColumnRenamed("week_2", "avg_processing_time_w2")
        .withColumnRenamed("week_3", "avg_processing_time_w3")
        .withColumnRenamed("week_4", "avg_processing_time_w4")
        ######################################## end
    )

    df_invoice_agg = (
        df_invoice_agg.withColumn("payment_date", F.add_months(occured_date_col, F.col("lag")))
        .join(df_processing_time, on=["payment_date"] + processing_seg_level, how="left")
        .drop("payment_date")
        .fillna(
            {
                "avg_processing_time_w1": 0.0,
                "avg_processing_time_w2": 0.0,
                "avg_processing_time_w3": 0.0,
                "avg_processing_time_w4": 0.0,
            }
        )
    )

    # Aggregating Preauth
    preauth_occured_date = "TreatmentEffectiveDate"
    common_segmentation = [c for c in df_preauth.columns if c in segmentation_features]
    df_preauth_agg = df_preauth.groupBy(preauth_occured_date, *common_segmentation).agg(F.count(preauth_occured_date).alias("nb_preauth"))

    # Invoice x Preauth
    df_invoice_agg = (
        df_invoice_agg.join(df_preauth_agg.withColumnRenamed(preauth_occured_date, occured_date_col), on=[occured_date_col] + common_segmentation, how="left")
        .fillna({"nb_preauth": 0})
        # Adding seasonnality feature
        .withColumn(occured_date_col + "_month", F.month(F.to_date(occured_date_col)))
        # Filtering
        .where(F.col("lag") < max_lags)
    )

    return df_invoice_agg
