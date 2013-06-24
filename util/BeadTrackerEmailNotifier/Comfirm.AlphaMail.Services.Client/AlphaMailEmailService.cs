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
using System.IO;
using System.Net;
using Comfirm.AlphaMail.Services.Client.Entities;
using Comfirm.AlphaMail.Services.Client.Entities.Internal;
using Comfirm.AlphaMail.Services.Client.Exceptions;
using Comfirm.Services.Client.Rest;
using Comfirm.Services.Client.Rest.Core;

namespace Comfirm.AlphaMail.Services.Client
{
    /// <summary>
    /// Service used to handle Transactional Emails with AlphaMail
    /// </summary>
    public class AlphaMailEmailService : IAlphaMailService, IEmailService
    {
        /// <summary>
        /// Url of the service. E.g. http://api.amail.io/v1/
        /// </summary>
        public Uri ServiceUrl { get; private set; }

        /// <summary>
        /// Token used for authentication
        /// </summary>
        public string ApiToken { get; private set; }

        /// <summary>
        /// Client used for making restful HTTP-requests
        /// </summary>
        private readonly Restful _client = new Restful();

        public AlphaMailEmailService() {}

        public AlphaMailEmailService(string serviceUrl, string apiToken)
        {
            this.SetServiceUrl(serviceUrl);
            this.SetApiToken(apiToken);
        }

        public AlphaMailEmailService(string apiToken)
        {
            this.SetServiceUrl("http://api.amail.io/v2/");
            this.SetApiToken(apiToken);
        }

        public AlphaMailEmailService SetServiceUrl(string serviceUrl)
        {
            Uri uriParseResult;

            if (!Uri.TryCreate(serviceUrl, UriKind.Absolute, out uriParseResult))
                throw new ArgumentException("Service url is not a properly formatted URI", "serviceUrl");

            this.ServiceUrl = uriParseResult;
            return this;
        }

        public AlphaMailEmailService SetApiToken(string apiToken)
        {
            this.ApiToken = apiToken;
            this._client.SetBasicAuthentication("", apiToken);
            return this;
        }

        public ServiceResponse<Guid?> Queue(IdempotentEmailMessagePayload message)
        {
            throw new NotImplementedException("Not implemented. But on our todo! Contact our support for more information.");
        }

        public ServiceResponse<Guid?> Queue(EmailMessagePayload payload)
        {
            var targetUrl = Path.Combine(this.ServiceUrl.OriginalString, "email/queue");
            var response = HandleErrors(() =>  this._client.Post<InternalServiceResponse<Guid?>, InternalEmailMessagePayload>(targetUrl, InternalEmailMessagePayload.Map(payload)));
            return InternalServiceResponse<Guid?>.Map(response.Result);
        }

        private static HttpResponse<InternalServiceResponse<TResult>> HandleErrors<TResult>(Func<HttpResponse<InternalServiceResponse<TResult>>> action)
        {
            HttpResponse<InternalServiceResponse<TResult>> response;

            try
            {
                response = action();
            }
            catch (Exception exception)
            {
                throw new AlphaMailServiceException("An unhandled exception occurred", exception);
            }

            switch (response.Head.Status.Code)
            {
                // Successful requests
                case HttpStatusCode.Accepted:
                case HttpStatusCode.Created:
                case HttpStatusCode.OK:
                    if (response.Result.error_code != 0)
                        throw new AlphaMailInternalException(string.Format(
                            "Service returned success while response error code was set ({0})", response.Result.error_code));
                    break;

                // Unauthorized
                case HttpStatusCode.Forbidden:
                case HttpStatusCode.Unauthorized:
                    throw new AlphaMailAuthorizationException(
                        response.Result.message,
                        response.Head.Status,
                        new ServiceResponse()
                        {
                            ErrorCode = response.Result.error_code,
                            Message = response.Result.message
                        },
                        null
                    );

                // Validation error
                case HttpStatusCode.MethodNotAllowed:
                case HttpStatusCode.BadRequest:
                    throw new AlphaMailValidationException(
                        response.Result.message,
                        response.Head.Status,
                        new ServiceResponse()
                        {
                            ErrorCode = response.Result.error_code,
                            Message = response.Result.message
                        },
                        null
                    );

                // Internal error
                case HttpStatusCode.InternalServerError:
                    throw new AlphaMailInternalException(
                        "An internal error occurred. Please contact our support for more information.",
                        response.Head.Status,
                        new ServiceResponse()
                        {
                            ErrorCode = response.Result.error_code,
                            Message = response.Result.message
                        },
                        null
                    );

                // Unknown
                default:
                    throw new AlphaMailServiceException(
                        "An unknown error occurred. Please contact our support for more information.",
                        response.Head.Status,
                        new ServiceResponse()
                        {
                            ErrorCode = response.Result.error_code,
                            Message = response.Result.message
                        },
                        null
                    );
            }

            return response;
        }
    }
}
