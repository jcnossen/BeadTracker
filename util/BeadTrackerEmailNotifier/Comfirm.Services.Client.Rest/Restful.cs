/*
The MIT License

Copyright (c) Robin Orheden, 2013 <http://amail.io/>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

using System;
using System.Text;
using Comfirm.Services.Client.Rest.Core;

namespace Comfirm.Services.Client.Rest
{
    /// <summary>
    /// Client used for making REST CRUD HTTP-operations
    /// </summary>
    public class Restful
    {
        /// <summary>
        /// Base http client used for making the REST-requests
        /// </summary>
        private readonly IHttpRequest _request;
        private readonly HttpHeaderCollection _headers;

        public Restful()
        {
            this._request = new HttpRequest();
            this._headers = new HttpHeaderCollection { { "Accept-Encoding", "gzip, deflate, chunked" } };
        }       

		public HttpResponse<TResult> Get<TResult>(string url)
            where TResult : class
		{
            return this._request.Request<TResult, object>(HttpMethodType.Get, url, this._headers, null);
		}

        public HttpResponse<TResult> Put<TResult, TBody>(string url, TBody body)
            where TResult : class
            where TBody : class
		{
            return this._request.Request<TResult, TBody>(HttpMethodType.Put, url, this._headers, body);
		}

        public HttpResponse<TResult> Post<TResult, TBody>(string url, TBody body)
            where TResult : class
            where TBody : class
		{
            return this._request.Request<TResult, TBody>(HttpMethodType.Post, url, this._headers, body);
		}

        public HttpResponse<TResult> Delete<TResult>(string url)
            where TResult : class
        {
            return this._request.Request<TResult, object>(HttpMethodType.Delete, url, this._headers, null);
        }

        public void SetBasicAuthentication(string username, string password)
		{
            if (this._headers.ContainsKey("Authorization"))
                this._headers.Remove("Autorization");
            
            this._headers.Add("Authorization", "Basic " +
                Convert.ToBase64String(Encoding.ASCII.GetBytes(string.Format("{0}:{1}", username, password))));
		}
    }
}
