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

using System.Text;
using System.IO;
using System.Runtime.Serialization.Json;
using System.Web.Script.Serialization;
using System;

namespace Comfirm.Services.Client.Rest.Core.Serialization
{
    public class JsonWebSerializer : IWebSerializer
    {
        public string ContentType { get { return "application/json"; } }
        public JavaScriptSerializer Serializer { get; private set; }

        public JsonWebSerializer()
        {
            this.Serializer = new JavaScriptSerializer();
        }

        public TObject Deserialize<TObject>(string source)
        {
            return this.Serializer.Deserialize<TObject>(source);
        }

        public string Serialize<TObject>(TObject source)
        {
            return this.Serializer.Serialize(source);
        }
    }
}
