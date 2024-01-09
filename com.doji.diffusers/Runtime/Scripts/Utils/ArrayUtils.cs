using System;

namespace Doji.AI.Diffusers {

    internal static class ArrayUtils {

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