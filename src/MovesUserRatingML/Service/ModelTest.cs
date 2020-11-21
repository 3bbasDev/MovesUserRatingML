using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MovesUserRatingML.DataDomain;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MovesUserRatingML.Service
{
    public static class ModelTest
    {

        public static string DatasetsRelativePath = @"../../../Files";


        private const float UserId = 191;
        private const int MoveId = 20;
        public static void M1()
        {
            string TrainData = GetAbsolutePath($"{DatasetsRelativePath}/recommendation-ratings-train.csv");
            string TestData = GetAbsolutePath($"{DatasetsRelativePath}/recommendation-ratings-test.csv");
            //Create instanse from ml
            MLContext ml = new MLContext();

            //load and map model from csv file
            IDataView dataView = ml.Data.LoadFromTextFile<MovieRating>(TrainData, hasHeader: true, separatorChar: ',');

            // Transfrom data to encode with train
            var DataProcessPipeLine = ml.Transforms.Conversion.MapValueToKey(outputColumnName: "userId", inputColumnName: nameof(MovieRating.userId))
                .Append(ml.Transforms.Conversion.MapValueToKey(outputColumnName: "movieId", inputColumnName: nameof(MovieRating.movieId)));

            // option of train model
            var Option = new MatrixFactorizationTrainer.Options();
            Option.LabelColumnName = "Label";
            Option.MatrixColumnIndexColumnName = "userId";
            Option.MatrixRowIndexColumnName = "movieId";
            Option.NumberOfIterations = 191;
            Option.ApproximationRank = 22;

            //add options to PipeLine
            var Train = DataProcessPipeLine.Append(ml.Recommendation().Trainers.MatrixFactorization(Option));

            //Train model to fitting to dataset 
            Console.WriteLine("=============== Training the model ===============");
            ITransformer model = Train.Fit(dataView);

            // Evaluate the model performance 
            Console.WriteLine("=============== Evaluating the model ===============");
            IDataView TestDataView = ml.Data.LoadFromTextFile<MovieRating>(TestData, hasHeader: true, separatorChar: ',');

            var Predaction = model.Transform(TestDataView);
            var metrics = ml.Regression.Evaluate(Predaction, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);

            //STEP 7:  Try/test a single prediction by predicting a single movie rating for a specific user
            var predictionengine = ml.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            /* Make a single movie rating prediction, the scores are for a particular user and will range from 1 - 5. 
               The higher the score the higher the likelyhood of a user liking a particular movie.
               You can recommend a movie to a user if say rating > 3.5.*/
            var movieratingprediction = predictionengine.Predict(
                new MovieRating()
                {
                    //Example rating prediction for userId = 6, movieId = 10 (Heat)
                    userId = UserId,
                    movieId = MoveId
                }
            );

            Movie movieService = new Movie(@"C:\Users\abbas\Downloads\CodeSample\Data\recommendation-movies.csv");
            Console.WriteLine("For userId:" + UserId + " movie rating prediction (1 - 5 stars) for movie:" + movieService.Get(MoveId).movieTitle + " is:" + Math.Round(movieratingprediction.Score, 1));

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
        }
        public static void M2()
        {
            string ModelPath = GetAbsolutePath($"{DatasetsRelativePath}/model.zip");
            string TrainingDataLocation = GetAbsolutePath($"{DatasetsRelativePath}/ratings-train.csv");
            string TestDataLocation = GetAbsolutePath($"{DatasetsRelativePath}/ratings-test.csv");

            Color color = Color.FromArgb(120, 100, 130);

            //Call the following piece of code for splitting the ratings.csv into ratings_train.csv and ratings.test.csv.
            // Program.DataPrep();

            //STEP 1: Create MLContext to be shared across the model creation workflow objects
            MLContext mlContext = new MLContext();

            //STEP 2: Read data from text file using TextLoader by defining the schema for reading the movie recommendation datasets and return dataview.
            var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating1>(path: TrainingDataLocation, hasHeader: true, separatorChar: ',');

            Console.WriteLine("=============== Reading Input Files ===============", color);
            Console.WriteLine();

            // ML.NET doesn't cache data set by default. Therefore, if one reads a data set from a file and accesses it many times, it can be slow due to
            // expensive featurization and disk operations. When the considered data can fit into memory, a solution is to cache the data in memory. Caching is especially
            // helpful when working with iterative algorithms which needs many data passes. Since SDCA is the case, we cache. Inserting a
            // cache step in a pipeline is also possible, please see the construction of pipeline below.
            trainingDataView = mlContext.Data.Cache(trainingDataView);

            Console.WriteLine("=============== Transform Data And Preview ===============", color);
            Console.WriteLine();

            //STEP 4: Transform your data by encoding the two features userId and movieID.
            //        These encoded features will be provided as input to FieldAwareFactorizationMachine learner
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "userIdFeaturized", inputColumnName: nameof(MovieRating1.userId))
                                          .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "movieIdFeaturized", inputColumnName: nameof(MovieRating1.movieId))
                                          .Append(mlContext.Transforms.Concatenate("Features", "userIdFeaturized", "movieIdFeaturized")));
            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 500);

            // STEP 5: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============", color);
            Console.WriteLine();
            var trainingPipeLine = dataProcessPipeline.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(new string[] { "Features" }));
            var model = trainingPipeLine.Fit(trainingDataView);

            //STEP 6: Evaluate the model performance
            Console.WriteLine("=============== Evaluating the model ===============", color);
            Console.WriteLine();
            var testDataView = mlContext.Data.LoadFromTextFile<MovieRating1>(path: TestDataLocation, hasHeader: true, separatorChar: ',');

            var prediction = model.Transform(testDataView);

            var metrics = mlContext.BinaryClassification.Evaluate(data: prediction, labelColumnName: "Label", scoreColumnName: "Score", predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine("Evaluation Metrics: acc:" + Math.Round(metrics.Accuracy, 2) + " AreaUnderRocCurve(AUC):" + Math.Round(metrics.AreaUnderRocCurve, 2), color);

            //STEP 7:  Try/test a single prediction by predicting a single movie rating for a specific user
            Console.WriteLine("=============== Test a single prediction ===============", color);
            Console.WriteLine();
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating1, MovieRatingPrediction1>(model);
            MovieRating1 testData = new MovieRating1() { userId = "6", movieId = "15" };

            var movieRatingPrediction = predictionEngine.Predict(testData);
            Console.WriteLine($"UserId:{testData.userId} with movieId: {testData.movieId} Score:{Sigmoid(movieRatingPrediction.Score)} and Label {movieRatingPrediction.PredictedLabel}", Color.YellowGreen);
            Console.WriteLine();

            //STEP 8:  Save model to disk
            Console.WriteLine("=============== Writing model to the disk ===============", color);
            Console.WriteLine(); mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);

            Console.WriteLine("=============== Re-Loading model from the disk ===============", color);
            Console.WriteLine();
            ITransformer trainedModel;
            using (FileStream stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }


            Console.WriteLine("Press any key ...");
            Console.Read();
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void DataPrep()
        {

            string[] dataset = File.ReadAllLines(@".\Files\ratings.csv");

            string[] new_dataset = new string[dataset.Length];
            new_dataset[0] = dataset[0];
            for (int i = 1; i < dataset.Length; i++)
            {
                string line = dataset[i];
                string[] lineSplit = line.Split(',');
                double rating = Double.Parse(lineSplit[2]);
                rating = rating > 3 ? 1 : 0;
                lineSplit[2] = rating.ToString();
                string new_line = string.Join(',', lineSplit);
                new_dataset[i] = new_line;
            }
            dataset = new_dataset;
            int numLines = dataset.Length;
            var body = dataset.Skip(1);
            var sorted = body.Select(line => new { SortKey = Int32.Parse(line.Split(',')[3]), Line = line })
                             .OrderBy(x => x.SortKey)
                             .Select(x => x.Line);
            File.WriteAllLines(@"../../../Files\ratings-train.csv", dataset.Take(1).Concat(sorted.Take((int)(numLines * 0.9))));
            File.WriteAllLines(@"../../../Files\ratings-test.csv", dataset.Take(1).Concat(sorted.TakeLast((int)(numLines * 0.1))));
        }

        public static float Sigmoid(float x)
        {
            return (float)(100 / (1 + Math.Exp(-x)));
        }
    }
    public class MovieRating1
    {
        [LoadColumn(0)]
        public string userId;

        [LoadColumn(1)]
        public string movieId;

        [LoadColumn(2)]
        public bool Label;
    }
    public class MovieRatingPrediction1
    {
        public bool PredictedLabel;

        public float Score;
    }
}
