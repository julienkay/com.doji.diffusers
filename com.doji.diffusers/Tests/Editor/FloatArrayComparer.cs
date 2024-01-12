using System;
using System.Collections;
using UnityEngine.TestTools.Utils;

namespace Doji.AI.Diffusers.Editor.Tests {

    /// <summary>
    /// Custom comparer for float arrays with tolerance
    /// </summary>
    public class FloatArrayComparer : IComparer {
        private readonly float _allowedError;

        public FloatArrayComparer(float allowedError) {
            _allowedError = allowedError;
        }

        public int Compare(object x, object y) {
            if (x is float floatX && y is float floatY) {
                if (Utils.AreFloatsEqual(floatX, floatY, _allowedError)) {
                    return 0;
                }
                return floatX.CompareTo(floatY);
            }
            throw new ArgumentException("Arguments must be of type float");
        }
    }
}