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
using System.Net;
using System.Text;

namespace Comfirm.Services.Client.Rest.Core
{
    public class ExtendedWebClient : WebClient
    {
        public Version StatusVersion
        {
            get
            {
                this.ReadStatusHeader();
                return this._statusVersion;
            }
        }

        public HttpStatusCode StatusCode
        {
            get
            {
                this.ReadStatusHeader();
                return this._statusCode;
            }
        }

        public string StatusDescription
        {
            get
            {
                this.ReadStatusHeader();
                return this._statusDescription;
            }
        }

        private WebRequest _request;
        private volatile bool _isHeaderRead;

        private Version _statusVersion;
        private HttpStatusCode _statusCode;
        private string _statusDescription;

        public ExtendedWebClient(Encoding encoding = null)
        {
            this.Encoding = encoding ?? Encoding.ASCII;
        }

        protected override WebRequest GetWebRequest(Uri address)
        {
            this._request = base.GetWebRequest(address);

            if (this._request is HttpWebRequest)
            {
                ((HttpWebRequest)this._request).AllowAutoRedirect = false;
                ((HttpWebRequest)this._request).KeepAlive = false;
            }

            return this._request;
        }

        private void ReadStatusHeader()
        {
            if (this._isHeaderRead)
                return;
            else
                _isHeaderRead = true;

            if (this._request == null)
                throw new InvalidOperationException("Unable to retrieve the status code, maybe you haven't made a request yet.");

            var response = base.GetWebResponse(this._request) as HttpWebResponse;

            if (response == null)
                throw new InvalidOperationException("Unable to retrieve the status code, maybe you haven't made a request yet.");

            this._statusVersion = response.ProtocolVersion;
            this._statusCode = response.StatusCode;
            this._statusDescription = response.StatusDescription;
        }
    }
}
