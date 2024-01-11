using System;

namespace Doji.AI.Diffusers {

    internal static class ArrayUtils {

        /// <summary>
        /// Return samples from the �standard normal� distribution.
        /// (Gaussian distribution of mean 0 and variance 1.)
        /// </summary>
        public static float[] Randn(int size, double mean = 0, double stdDev = 1) {
            Random random = new Random();
            float[] randomArray = new float[size];

            for (int i = 0; i < size; i++) {
                randomArray[i] = (float)random.SampleGaussian(mean, stdDev);
            }

            return randomArray;
        }

        private static double SampleGaussian(this Random random, double mean = 0, double stdDev = 1) {
            double u1 = 1 - random.NextDouble();
            double u2 = 1 - random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            return z * stdDev + mean;
        }

        public static T[] Concat<T>(this T[] array1, T[] array2) {
            if (array1 == null) {
                throw new ArgumentNullException(nameof(array1));
            }

            if (array2 == null) {
                throw new ArgumentNullException(nameof(array2));
            }

            T[] resultArray = new T[array1.Length + array2.Length];
            Array.Copy(array1, resultArray, array1.Length);
            Array.Copy(array2, 0, resultArray, array1.Length, array2.Length);
            return resultArray;
        }

        /// <summary>
        /// Takes a array, repeats it twice and returns the repeated sequence as a new array.
        /// </summary>
        public static T[] Repeat<T>(this T[] array) {
            if (array == null) {
                throw new ArgumentNullException(nameof(array));
            }
            T[] repeatedArray = new T[array.Length * 2];
            Array.Copy(array, 0, repeatedArray, 0, array.Length);
            Array.Copy(array, 0, repeatedArray, array.Length, array.Length);
            return repeatedArray;
        }

        /// <summary>
        /// numpy.full
        /// </summary>
        public static int[] Full(this int n, int x) {
            if (n < 0) {
                throw new ArgumentException("Value of n must be non-negative.");
            }

            int[] array = new int[n];

            for (int i = 0; i < n; i++) {
                array[i] = x;
            }

            return array;
        }

        /// <summary>
        /// numpy.cumprod
        /// </summary>
        public static float[] CumProd(this float[] array) {
            int length = array.Length;
            float[] result = new float[length];
            float product = 1.0f;

            for (int i = 0; i < length; i++) {
                product *= array[i];
                result[i] = product;
            }

            return result;
        }

        /// <summary>
        /// numpy.arange
        /// </summary>
        public static int[] Arange(int start, int stop, int step = 1) {
            if (step <= 0) {
                throw new ArgumentException("Step must be a positive integer.");
            }

            int length = ((stop - start - 1) / step) + 1;
            int[] result = new int[length];

            for (int i = 0, value = start; i < length; i++, value += step) {
                result[i] = value;
            }

            return result;
        }

        /// <summary>
        /// numpy.linspace
        /// </summary>
        public static float[] Linspace(float start, float stop, int num) {
            if (num <= 1) {
                throw new ArgumentException("Number of elements must be greater than 1.");
            }

            float[] result = new float[num];
            float step = (stop - start) / (float)(num - 1);

            for (int i = 0; i < num; i++) {
                result[i] = start + i * step;
            }

            return result;
        }
    }
}