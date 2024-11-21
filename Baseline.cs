using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BaselineEvaluation
{
    // Interfaces (unchanged)
    public interface IEvaluator
    {
        double EvaluatePredictions(IEnumerable<(IDictionary<string, double>, string)> labeledExamples, IEnumerable<string> predictions);
    }

    public interface ITreeBuilder
    {
        IDecisionTree BuildTree(IEnumerable<(IDictionary<string, double>, string)> labeledExamples);
    }

    public interface IEvalHelper
    {
        IEnumerable<string> MakePredictions(ILabeler labeler, IEnumerable<IDictionary<string, double>> unlabeledExamples);
        IEnumerable<string> MakeNFoldCrossPredictions(ITreeBuilder builder, IEnumerable<(IDictionary<string, double>, string)> labeledExamples);
    }

    // Models (unchanged)
    public interface IDecisionTree
    {
        string Label(IDictionary<string, double> features);
    }

    public class DecisionTree : IDecisionTree
    {
        private readonly string _mostFrequentLabel;

        public DecisionTree(string mostFrequentLabel)
        {
            _mostFrequentLabel = mostFrequentLabel;
        }

        public string Label(IDictionary<string, double> features)
        {
            return _mostFrequentLabel;
        }
    }

    public interface ILabeler
    {
        string Label(IDictionary<string, double> features);
    }

    public class TreeLabeler : ILabeler
    {
        private readonly IDecisionTree _tree;

        public TreeLabeler(IDecisionTree tree)
        {
            _tree = tree;
        }

        public string Label(IDictionary<string, double> features)
        {
            return _tree.Label(features);
        }
    }

    // CSV Helper Class
    public static class CSVHelper
    {
        public static List<(IDictionary<string, double>, string)> LoadLabeledData(string filePath, string labelColumn)
        {
            var labeledData = new List<(IDictionary<string, double>, string)>();
            var lines = File.ReadAllLines(filePath);

            // Assumes first row contains headers
            var headers = lines[0].Split(',');
            int labelIndex = Array.IndexOf(headers, labelColumn);

            if (labelIndex == -1)
                throw new ArgumentException($"Label column '{labelColumn}' not found in CSV headers.");

            for (int i = 1; i < lines.Length; i++)
            {
                var row = lines[i].Split(',');
                var features = new Dictionary<string, double>();

                for (int j = 0; j < headers.Length; j++)
                {
                    if (j != labelIndex) // Exclude label column
                        features[headers[j]] = double.Parse(row[j]);
                }

                labeledData.Add((features, row[labelIndex]));
            }

            return labeledData;
        }

        public static List<IDictionary<string, double>> LoadUnlabeledData(string filePath)
        {
            var unlabeledData = new List<IDictionary<string, double>>();
            var lines = File.ReadAllLines(filePath);

            // Assumes first row contains headers
            var headers = lines[0].Split(',');

            for (int i = 1; i < lines.Length; i++)
            {
                var row = lines[i].Split(',');
                var features = new Dictionary<string, double>();

                for (int j = 0; j < headers.Length; j++)
                {
                    features[headers[j]] = double.Parse(row[j]);
                }

                unlabeledData.Add(features);
            }

            return unlabeledData;
        }
    }

    // Implementations (unchanged)
    public class AccuracyEvaluator : IEvaluator
    {
        public double EvaluatePredictions(IEnumerable<(IDictionary<string, double>, string)> labeledExamples, IEnumerable<string> predictions)
        {
            var labeledList = labeledExamples.ToList();
            var predictionList = predictions.ToList();

            if (labeledList.Count != predictionList.Count)
                throw new ArgumentException("Mismatch between labeled examples and predictions count.");

            int correct = labeledList.Zip(predictionList, (example, prediction) => example.Item2 == prediction).Count(match => match);

            return (double)correct / labeledList.Count;
        }
    }

    public class BaselineTreeBuilder : ITreeBuilder
    {
        public IDecisionTree BuildTree(IEnumerable<(IDictionary<string, double>, string)> labeledExamples)
        {
            var mostFrequentLabel = labeledExamples
                .GroupBy(example => example.Item2)
                .OrderByDescending(group => group.Count())
                .First()
                .Key;

            return new DecisionTree(mostFrequentLabel);
        }
    }

    public class EvalHelper : IEvalHelper
    {
        public IEnumerable<string> MakePredictions(ILabeler labeler, IEnumerable<IDictionary<string, double>> unlabeledExamples)
        {
            foreach (var example in unlabeledExamples)
            {
                yield return labeler.Label(example);
            }
        }

        public IEnumerable<string> MakeNFoldCrossPredictions(ITreeBuilder builder, IEnumerable<(IDictionary<string, double>, string)> labeledExamples)
        {
            var labeledList = labeledExamples.ToList();
            int foldSize = labeledList.Count / 5;

            for (int i = 0; i < 5; i++)
            {
                var testSet = labeledList.Skip(i * foldSize).Take(foldSize).ToList();
                var trainingSet = labeledList.Except(testSet).ToList();

                var tree = builder.BuildTree(trainingSet);
                var labeler = new TreeLabeler(tree);

                foreach (var example in testSet)
                {
                    yield return labeler.Label(example.Item1);
                }
            }
        }
    }

    // Main Program
    class Program
    {
        static void Main(string[] args)
        {
            // Load data from CSV
            var labeledData = CSVHelper.LoadLabeledData("data.csv", "LabelColumn");
            var unlabeledData = CSVHelper.LoadUnlabeledData("data.csv");

            // Tree Builder and Evaluator
            ITreeBuilder builder = new BaselineTreeBuilder();
            IEvaluator evaluator = new AccuracyEvaluator();
            IEvalHelper evalHelper = new EvalHelper();

            // Build Tree and Make Predictions
            var tree = builder.BuildTree(labeledData);
            var labeler = new TreeLabeler(tree);

            var predictions = evalHelper.MakePredictions(labeler, unlabeledData);
            var trainingAccuracy = evaluator.EvaluatePredictions(labeledData, predictions);

            Console.WriteLine($"Training Accuracy: {trainingAccuracy:F2}");

            // Perform Cross-Validation
            var crossPredictions = evalHelper.MakeNFoldCrossPredictions(builder, labeledData);
            var crossValidationAccuracy = evaluator.EvaluatePredictions(labeledData, crossPredictions);

            Console.WriteLine($"Cross-Validation Accuracy: {crossValidationAccuracy:F2}");
        }
    }
}
