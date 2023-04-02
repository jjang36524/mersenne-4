#include <smmintrin.h>
#include <immintrin.h>
#include <bits/stdc++.h>
using namespace std;
#define int long long
#pragma GCC target("avx2")
#pragma GCC target("fma")
__m256d mult(__m256d a, __m256d b) {
	__m256d c = _mm256_movedup_pd(a);
	__m256d d = _mm256_shuffle_pd(a, a, 15);
	__m256d cb = _mm256_mul_pd(c, b);
	__m256d db = _mm256_mul_pd(d, b);
	__m256d e = _mm256_shuffle_pd(db, db, 5);
	__m256d r = _mm256_addsub_pd(cb, e);
	return r;
}
void fft(int n, __m128d a[], bool invert) {
	for (int i = 1, j = 0; i < n; ++i) {
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(a[i], a[j]);
	}
	for (int len = 2; len <= n; len <<= 1) {
		double ang = 2 * 3.14159265358979 / len * (invert ? -1 : 1);
		__m256d wlen; wlen.m256d_f64[0] = cos(ang), wlen.m256d_f64[1] = sin(ang);
		for (int i = 0; i < n; i += len) {
			__m256d w; w.m256d_f64[0] = 1; w.m256d_f64[1] = 0;
			for (int j = 0; j < len / 2; ++j) {
				w = _mm256_permute2f128_pd(w, w, 0);
				wlen = _mm256_insertf128_pd(wlen, a[i + j + len / 2], 1);
				w = mult(w, wlen);
				__m128d vw = _mm256_extractf128_pd(w, 1);
				__m128d u = a[i + j];
				a[i + j] = _mm_add_pd(u, vw);
				a[i + j + len / 2] = _mm_sub_pd(u, vw);
			}
		}
	}
	if (invert) {
		__m128d inv; inv.m128d_f64[0] = inv.m128d_f64[1] = 1.0 / n;
		for (int i = 0; i < n; ++i) a[i] = _mm_mul_pd(a[i], inv);
	}
}
vector<int> multiply(vector<int>& v, vector<int>& w) {
	int n = 2; while (n < v.size() + w.size()) n <<= 1;
	__m128d* fv = new __m128d[n];
	for (int i = 0; i < n; ++i) fv[i].m128d_f64[0] = fv[i].m128d_f64[1] = 0;
	for (int i = 0; i < v.size(); ++i) fv[i].m128d_f64[0] = v[i];
	for (int i = 0; i < w.size(); ++i) fv[i].m128d_f64[1] = w[i];
	fft(n, fv, 0); // (a+bi) is stored in FFT
	for (int i = 0; i < n; i += 2) {
		__m256d a;
		a = _mm256_setzero_pd();
		a = _mm256_insertf128_pd(a, fv[i], 0);
		a = _mm256_insertf128_pd(a, fv[i + 1], 1);
		a = mult(a, a);
		fv[i] = _mm256_extractf128_pd(a, 0);
		fv[i + 1] = _mm256_extractf128_pd(a, 1);
	}
	fft(n, fv, 1);
	vector<int> ret(n);
	for (int i = 0; i < n; ++i) ret[i] = (int)round(fv[i].m128d_f64[1] / 2);
	delete[] fv;
	return ret;
}
struct bs {
	vector<int>arr;
	bs() {

	}
	bs(int n) {
		arr.resize((n >> 3) + 1);
	}
	void setnb(int n, int k)
	{
		arr[n >> 3] &= (255 - (1 << (n & 7)));
		if (k)
		{
			arr[n >> 3] += (1 << (n & 7));
		}
	}
	int getnb(int n)
	{
		return !!(arr[n >> 3] & (1 << (n & 7)));
	}
	void mod(int n)
	{
		int s = arr.size();
		int i;
		for (i = s - 1; i >= 0; i--)
		{
			int j;
			for (j = 7; j >= 0; j--)
			{
				int v = i * 7 + j;
				if (v < n)
					goto T;
				if (getnb(v))
				{
					setnb(v, 0);
					v -= n;
					while (getnb(v))
					{
						setnb(v, 0);
						v++;
					}
					setnb(v, 1);
				}
			}
		}
	T:
		int r = n - 1;
		for (i = r; i >= 0; i--)
		{
			if (getnb(i) == 0)
				break;
		}
		if (i < 0)
		{
			for (i = r; i >= 0; i--)
			{
				setnb(i, 0);
			}
		}
	}
};
bs mult(bs a, bs b, int n)
{
	auto r = multiply(a.arr, b.arr);
	r.resize(r.size() + 5);
	int i;
	for (i = 0; i < r.size() - 1; i++)
	{
		r[i + 1] += r[i] >> 8;
		r[i] &= 255;
	}
	bs t;
	t.arr = r;
	t.mod(n);
	t.arr.resize((n / 8) + 1);
	return t;
}
bs min2(bs a, int n)
{
	int i;
	a.arr[0] += 1;
	for (i = 2; i < n; i++)
	{
		a.arr[i >> 3] += (1 << (i & 7));
	}
	a.arr.resize(a.arr.size() + 5);
	for (i = 0; i < a.arr.size() - 1; i++)
	{
		a.arr[i + 1] += a.arr[i] >> 8;
		a.arr[i] &= 255;
	}
	a.mod(n);
	a.arr.resize((n / 8) + 1);
	return a;
}
signed main()
{
	while (1)
	{
		int N;
		cin >> N;
		auto start = chrono::high_resolution_clock::now();
		int i;
		bs an(N);
		an.arr[0] = 4;
		for (i = 0; i < N - 2; i++)
		{
			an = mult(an, an, N);
			an = min2(an, N);
			if (i % 1000 == 0)
				cout << i << '\n';
		}
		for (i = 0; i < 5; i++)
			cout << an.arr[i] << ' ';
		cout << '\n';
		auto finish = chrono::high_resolution_clock::now();
		cout << chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / (1000000.0) << "밀리초\n";
	}
	
}