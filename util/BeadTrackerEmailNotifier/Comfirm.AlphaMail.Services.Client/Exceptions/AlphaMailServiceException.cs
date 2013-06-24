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
using Comfirm.AlphaMail.Services.Client.Entities;
using Comfirm.Services.Client.Rest.Core;

namespace Comfirm.AlphaMail.Services.Client.Exceptions
{
    public class AlphaMailServiceException : Exception
    {
        public HttpResponseStatusHead HttpStatus { get; private set; }

        public ServiceResponse Response { get; private set; }

        public AlphaMailServiceException()
            : base() {}

        public AlphaMailServiceException(string message)
            : base(message) { }
        
        public AlphaMailServiceException(string message, Exception innerException)
            : base(message, innerException) {}

        public AlphaMailServiceException(string message, HttpResponseStatusHead httpStatus, ServiceResponse response, Exception innerException)
            : base(message, innerException)
        {
            this.HttpStatus = httpStatus;
            this.Response = response;
        }

        public int? GetErrorCode()
        {
            return this.Response == null ? null : new int?(this.Response.ErrorCode);
        }
    }
}