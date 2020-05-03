"""
This module performs statistical analysis on the noval corona virus dataset. The dataset being used 
was last updated on May 02, 2020. The Module performs the following Functions:
* Displays the statistics of input dataset
* Reads data from csv files and stores the aggregated output in parquet format
* Counts the Number of records for each country/region and provice/state
* Lists max Cases for each country/region and provice/state
* Lists max Deaths for each country/region and provice/state
* List max Recoveries for each country/region and provice/state
*

"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as func
import config as cfg


def data_cleansing(raw_data_df):
    """
    This method cleanses the data before further processing. Drop rows if having significant n/a values or
    fills the n/a values with a custom value.
    Provice/State column is filled with a custom value of 'Province/State' for all rows missing this value.

    :param raw_data_df: dataframe having raw data.
    :return : dataframe with cleansed data.
    """
    corona_df = raw_data_df.dropna(thresh=4)
    cleansed_data_df = corona_df.fillna(
        value='Province/State', subset='Province/State')
    return cleansed_data_df


def count_records_by_country_region(dataframe):
    """
    Groups the records by Country/Region column, counts and orders rows by count in descending order.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    return dataframe.groupBy('Country/Region').count().orderBy('count', ascending=False)


def count_records_by_province_state(dataframe):
    """
    Groups the records by Province/State column, counts and orders rows by count in descending order.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    return dataframe.groupBy('Province/State').count().orderBy('count', ascending=False)


def most_cases_by_country_region(dataframe):
    """
    Groups the records by Country/Region column, aggregates with max(Confirmed) column and sorts them in
    descending order of max of Confirmed cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy('Country/Region').max('Confirmed').select(
        'Country/Region', func.col("max(Confirmed)").alias("Most_Cases"))
    return result_df.orderBy('max(Confirmed)', ascending=False)


def most_cases_by_province_state(dataframe):
    """
    Groups the records by Province/State and Country/Region column, aggregates with max(Confirmed) column and sorts them in
    descending order of max of Confirmed cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy(
        'Province/State', 'Country/Region').max('Confirmed').select(
        'Province/State', 'Country/Region', func.col("max(Confirmed)").alias("Most_Cases"))
    return result_df.orderBy('max(Confirmed)', ascending=False)


def most_deaths_by_country_region(dataframe):
    """
    Groups the records by Country/Region column, aggregates with max(Deaths) column and sorts them in
    descending order of max of Deaths cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy('Country/Region').max('Deaths').select(
        'Country/Region', func.col("max(Deaths)").alias("Most_Deaths"))
    return result_df.orderBy('max(Deaths)', ascending=False)


def most_deaths_by_province_state(dataframe):
    """
    Groups the records by Province/State and Country/Region column column, aggregates with max(Deaths) column and sorts them in
    descending order of max of Deaths cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy(
        'Province/State', 'Country/Region').max('Deaths').select(
        'Province/State', 'Country/Region', func.col("max(Deaths)").alias("Most_Deaths"))
    return result_df.orderBy('max(Deaths)', ascending=False)


def most_recoveries_by_country_region(dataframe):
    """
    Groups the records by Country/Region column, aggregates with max(Recovered) column and sorts them in
    descending order of max of Recovered cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy('Country/Region').max('Recovered').select(
        'Country/Region', func.col("max(Recovered)").alias("Most_Recovered"))
    return result_df.orderBy('max(Recovered)', ascending=False)


def most_recoveries_by_province_state(dataframe):
    """
    Groups the records by Province/State and Country/Region column column, aggregates with max(Recovered) column and sorts them in
    descending order of max of Recovered cases.

    :param dataframe : dataframe having cleansed data
    :return : dataframe with aggregated data
    """
    result_df = dataframe.groupBy(
        'Province/State', 'Country/Region').max('Recovered').select(
        'Province/State','Country/Region', func.col("max(Recovered)").alias("Most_Recovered"))
    return result_df.orderBy('max(Recovered)', ascending=False)


def save_data_in_parquet(filename, dataframe):
    """
    This method saves the resultant aggregated data in parquet format.

    :param filename: name of the output file
    :param dataframe: the dataframe containing the aggregated data
    :return : none

    """
    dataframe.coalesce(1).write.mode('overwrite').parquet(filename)


def dispaly_dataset_info(csv_df):
    """
    Helper function that prints the schema of given file, first 20 rows, and statistics of the dataset.

    :param csv_df: the dataframe into which csv data is uploaded
    :return :none
    """

    # prints schema of the given file
    csv_df.printSchema()

    # prints the first 20 rows of the given file
    csv_df.show()

    # summary of attributes in the given file
    csv_df.describe().show()


def dispaly(dataframe):
    """
    Helper function that prints first n rows as set in config.py file that are in the dataframe.

    :param dataframe: the dataframe that is to be displayed.
    :return : none.
    """
    dataframe.show(n=cfg.ROW_NUM)


if __name__ == '__main__':

    # create spark session with given App name, or gets an already existing one
    spark = SparkSession.builder.appName(cfg.APP_NAME).getOrCreate()
    corona_raw_df = spark.read.csv(
        cfg.FILE_PATH, inferSchema=True, header=True)

    # display dataset statistics, schema and top rows.
    dispaly_dataset_info(corona_raw_df)

    # drop rows with significant n/a values otherwise fill with custom value
    corona_filtered_df = data_cleansing(corona_raw_df)

    # count of records by country or region
    output_file = 'records_count_by_country_region'
    cases_country_region_df = count_records_by_country_region(
        corona_filtered_df)
    save_data_in_parquet(output_file, cases_country_region_df)
    dispaly(cases_country_region_df)

    # count of records by province or state
    output_file = 'records_count_by_province_state'
    cases_province_state_df = count_records_by_province_state(
        corona_filtered_df)
    save_data_in_parquet(output_file, cases_country_region_df)
    dispaly(cases_province_state_df)

    # most Confirmed cases by country or region
    output_file = 'confirmed_cases_by_country_region'
    most_cases_country_region_df = most_cases_by_country_region(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_cases_country_region_df)
    dispaly(most_cases_country_region_df)

    # most Confirmed cases by province or state
    output_file = 'confirmed_cases_by_province_state'
    most_cases_province_state_df = most_cases_by_province_state(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_cases_province_state_df)
    dispaly(most_cases_province_state_df)

    # most Deaths by country or region
    output_file = 'deaths_count_by_country_region'
    most_deaths_country_region_df = most_deaths_by_country_region(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_deaths_country_region_df)
    dispaly(most_deaths_country_region_df)

    # most Deaths by province or state of a country or region
    output_file = 'deaths_count_by_province_state'
    most_deaths_province_state_df = most_deaths_by_province_state(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_deaths_province_state_df)
    dispaly(most_deaths_province_state_df)

    # most Recoveries by country or region
    output_file = 'recovery_count_by_country_region'
    most_recoveries_country_region_df = most_recoveries_by_country_region(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_recoveries_country_region_df)
    dispaly(most_recoveries_country_region_df)

    # most Recoveries by province or state of a country or region
    output_file = 'recovery_count_by_province_state'
    most_recoveries_province_state_df = most_recoveries_by_province_state(
        corona_filtered_df)
    save_data_in_parquet(output_file, most_recoveries_province_state_df)
    dispaly(most_recoveries_province_state_df)

    # spark.stop()
