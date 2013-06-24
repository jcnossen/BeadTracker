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
using System.Linq;
using System.Text;
using System.Net;
using System.IO;
using Comfirm.Services.Client.Rest.Core.Serialization;

namespace Comfirm.Services.Client.Rest.Core
{
    public class HttpRequest : IHttpRequest
    {
        private readonly IWebSerializer _serializer;

        public HttpRequest()
        {
            this._serializer = new JsonWebSerializer();
        }

        public HttpResponse<string> RequestRaw(HttpMethodType method, string url, HttpHeaderCollection headers = null)
        {
            return this.InternalRequest<string, object>(method, url, headers, null);
        }

        public HttpResponse<string> RequestRaw<TBody>(HttpMethodType method, string url, HttpHeaderCollection headers = null, TBody body = null)
            where TBody : class
        {
            return this.InternalRequest<string, TBody>(method, url, headers, body);
        }

        public HttpResponse<TResult> Request<TResult, TBody>(HttpMethodType method, string url, HttpHeaderCollection headers = null, TBody body = null)
            where TResult : class
            where TBody : class
        {
            return this.InternalRequest<TResult, TBody>(method, url, headers, body);
        }

        private HttpResponse<TResult> InternalRequest<TResult, TBody>(HttpMethodType method, string url, HttpHeaderCollection headers, TBody body)
            where TResult : class
            where TBody : class
        {
            TResult result;
            string rawResult;

            var head = new HttpResponseHead();
            var client = new ExtendedWebClient(Encoding.UTF8);
            headers = headers ?? new HttpHeaderCollection();

            headers.ToList().ForEach(x => client.Headers.Add(x.Key, x.Value));

            try
            {
                switch (method)
                {
                    case HttpMethodType.Get:
                        rawResult = client.DownloadString(url);
                        break;

                    case HttpMethodType.Post:
                        if (body == null) throw new NullReferenceException("Body cannot be NULL when performing POST");
                        client.Headers.Add("Content-Type", this._serializer.ContentType);
                        rawResult = client.UploadString(url, "POST", this._serializer.Serialize<TBody>(body));
                        break;

                    case HttpMethodType.Put:
                        if (body == null) throw new NullReferenceException("Body cannot be NULL when performing PUT");
                        client.Headers.Add("Content-Type", this._serializer.ContentType);
                        rawResult = client.UploadString(url, "PUT", this._serializer.Serialize<TBody>(body));
                        break;

                    case HttpMethodType.Delete:
                        rawResult = client.UploadString(url, "DELETE", "");
                        break;

                    default:
                        throw new InvalidOperationException("Invalid http method type specified");
                }

                head.Status = new HttpResponseStatusHead()
                {
                    Code = client.StatusCode,
                    Message = client.StatusDescription,
                    Version = client.StatusVersion.Major == 1 && client.StatusVersion.Minor == 1 ?
                        HttpProtocolVersionType.Http11 : HttpProtocolVersionType.Http10
                };

                result = typeof(TResult) == typeof(string) ? rawResult as TResult
                    : (TResult)this._serializer.Deserialize<TResult>(rawResult);
            }
            catch (WebException exception)
            {
                if (exception.Response != null)
                {
                    var response = exception.Response as HttpWebResponse;

                    head.Status = new HttpResponseStatusHead()
                    {
                        Code = response.StatusCode,
                        Message = response.StatusDescription,
                        Version = response.ProtocolVersion.Major == 1 && response.ProtocolVersion.Minor == 1 ?
                            HttpProtocolVersionType.Http11 : HttpProtocolVersionType.Http10
                    };

                    rawResult = new StreamReader(exception.Response.GetResponseStream()).ReadToEnd();
                    result = typeof(TResult) == typeof(string) ? rawResult as TResult : (TResult)this._serializer.Deserialize<TResult>(rawResult);
                }
                else
                {
                    throw;
                }
            }
            finally
            {
                head.Headers = this.MapHeaderCollection(client.ResponseHeaders);
            }

            return new HttpResponse<TResult>() { Result = result, Head = head };
        }

        private HttpHeaderCollection MapHeaderCollection(WebHeaderCollection source)
        {
            var resultHeaders = new HttpHeaderCollection();

            if (source == null)
                return resultHeaders;

            foreach (var key in source.Keys)
                resultHeaders.Add((string)key, (string)source[(string)key]);

            return resultHeaders;
        }
    }
}
