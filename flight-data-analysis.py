from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType


# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------


def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate scheduled and actual travel times
    flights_df = flights_df.withColumn(
        "ScheduledTravelTime", F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")
    ).withColumn(
        "ActualTravelTime", F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")
    )

    # Calculate the discrepancy (absolute difference)
    flights_df = flights_df.withColumn("Discrepancy", F.abs(flights_df["ScheduledTravelTime"] - flights_df["ActualTravelTime"]))

    # Rank by largest discrepancy using window functions
    window_spec = Window.partitionBy("CarrierCode").orderBy(F.desc("Discrepancy"))
    
    flights_df = flights_df.withColumn("Rank", F.row_number().over(window_spec))

    # Join with carriers for carrier names
    flights_df = flights_df.join(carriers_df, on="CarrierCode", how="inner")

    # Select required columns and filter top results (you can adjust the rank if needed)
    result = flights_df.select(
        "FlightNum", "CarrierName", "Origin", "Destination",
        "ScheduledTravelTime", "ActualTravelTime", "Discrepancy"
    ).filter(flights_df["Rank"] <= 10)  # Adjust this to get top N flights

    result.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")


# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    # Calculate departure delay
    flights_df = flights_df.withColumn(
        "DepartureDelay", F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")
    )

    # Group by carrier and calculate standard deviation of departure delays
    carrier_delay_stats = flights_df.groupBy("CarrierCode").agg(
        F.count("FlightNum").alias("FlightCount"),
        F.stddev("DepartureDelay").alias("StdDevDepartureDelay")
    )

    # Filter carriers with more than 100 flights
    carrier_delay_stats = carrier_delay_stats.filter(carrier_delay_stats["FlightCount"] > 100)

    # Join with carrier names
    carrier_delay_stats = carrier_delay_stats.join(carriers_df, on="CarrierCode", how="inner")

    # Select and rank by standard deviation of departure delays
    result = carrier_delay_stats.select(
        "CarrierName", "FlightCount", "StdDevDepartureDelay"
    ).orderBy("StdDevDepartureDelay")

    result.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Step 1: Identify canceled flights where ActualDeparture is null
    canceled_flights = flights_df.filter(flights_df["ActualDeparture"].isNull())

    # Step 2: Group by origin and destination to count canceled and total flights
    canceled_counts = canceled_flights.groupBy("Origin", "Destination").agg(
        F.count("FlightNum").alias("CanceledFlights")
    )
    
    total_counts = flights_df.groupBy("Origin", "Destination").agg(
        F.count("FlightNum").alias("TotalFlights")
    )

    # Step 3: Join to calculate the cancellation rate
    route_cancellation = canceled_counts.join(
        total_counts,
        on=["Origin", "Destination"],
        how="inner"
    ).withColumn(
        "CancellationRate", F.col("CanceledFlights") / F.col("TotalFlights")
    )

    # Step 4: Alias the airports DataFrame for both origin and destination
    airports_origin = airports_df.select(
        airports_df["AirportCode"].alias("OriginCode"),
        airports_df["AirportName"].alias("OriginAirport"),
        airports_df["City"].alias("OriginCity")
    )

    airports_destination = airports_df.select(
        airports_df["AirportCode"].alias("DestinationCode"),
        airports_df["AirportName"].alias("DestinationAirport"),
        airports_df["City"].alias("DestinationCity")
    )

    # Step 5: Join with the airports DataFrame to get airport names and cities for both origin and destination
    route_cancellation = route_cancellation.join(
        airports_origin,
        route_cancellation["Origin"] == airports_origin["OriginCode"],
        how="left"
    ).join(
        airports_destination,
        route_cancellation["Destination"] == airports_destination["DestinationCode"],
        how="left"
    )

    # Step 6: Explicitly select columns to avoid ambiguity
    result = route_cancellation.select(
        F.col("OriginAirport"),
        F.col("OriginCity"),
        F.col("DestinationAirport"),
        F.col("DestinationCity"),
        F.col("CancellationRate")
    ).orderBy(F.desc("CancellationRate"))

    # Step 7: Write the result to a CSV file
    result.write.csv(task3_output, header=True, mode="overwrite")
    print(f"Task 3 output written to {task3_output}")


# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Create time of day group
    def time_of_day(hour):
        if 6 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 24:
            return "Evening"
        else:
            return "Night"

    time_of_day_udf = F.udf(time_of_day, StringType())
    flights_df = flights_df.withColumn("TimeOfDay", time_of_day_udf(F.hour("ScheduledDeparture")))

    # Calculate departure delay
    flights_df = flights_df.withColumn(
        "DepartureDelay", F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")
    )

    # Group by carrier and time of day to calculate average delay
    carrier_time_of_day = flights_df.groupBy("CarrierCode", "TimeOfDay").agg(
        F.avg("DepartureDelay").alias("AvgDepartureDelay")
    )

    # Join with carrier names
    carrier_time_of_day = carrier_time_of_day.join(carriers_df, on="CarrierCode", how="inner")

    result = carrier_time_of_day.select(
        "CarrierName", "TimeOfDay", "AvgDepartureDelay"
    ).orderBy("TimeOfDay", "AvgDepartureDelay")

    result.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
