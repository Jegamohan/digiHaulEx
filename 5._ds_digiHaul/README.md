#Tasks

##1. On-Time Delivery Performance Analysis

Operational teams rely on KPIs like on-time collection and on-time delivery to gauge carrier performance.

Question: What percentage of shipments met the on-time delivery threshold (arriving no later than 30 minutes past the scheduled delivery window) between October 1st and December 31st, 2023?

Assumptions:

* I have assumed that the time difference between LAST_DELIVERY_SCHEDULE_LATEST and the maximum GPS time determines if the delivery met the on-time delivery threshold.
* The percentage of Delivery_Status = 1 is 70.49% for October, compared to an overall percentage of Delivery_Status = 2 at 62.65%. This seems significantly high.
* At this point, it would be good to compare the monthly percentages of Delivery_Status to understand if there are any seasonal trends affecting our delivery performance.


Additionally, analyzing these trends could help identify specific periods or months where on-time delivery rates fluctuate, enabling us to implement targeted strategies for improvement.

Model Performance and Future Enhancements

My classification model is achieving an accuracy of about 70% with the current sets of data that I have. While that’s a good start, I believe there’s more data we can add to improve the accuracy.

##Data Sources to Consider

1. Weather Reports: Integrating weather data could provide valuable insights into how good or bad weather impacts delivery.

2. Traffic Updates: Adding real-time traffic data and using it to better plan routes will help improve the accuracy of the model and enhance route planning.

3. Historical Trends: By incorporating historical data, the model can learn from past patterns, which can significantly enhance its predictive accuracy.

By enriching the dataset with these additional features, I hope to not only improve the model’s accuracy but also make it more robust and effective overall.



#Technical Design for Deploying the Prediction Model (Using AWS SageMaker)

I will deploy the model in AWS SageMaker. To do this, I will first train my model in a SageMaker notebook and store the training data in an S3 bucket. I will create a Docker image that uses the trained model stored in the S3 location. Then, I will create an endpoint for the model and invoke it to obtain predictions.
