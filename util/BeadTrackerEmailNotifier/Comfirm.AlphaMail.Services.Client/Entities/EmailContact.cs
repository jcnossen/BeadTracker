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

namespace Comfirm.AlphaMail.Services.Client.Entities
{
    public class EmailContact
    {
        /// <summary>
        /// Id of the contact
        /// </summary>
        public string Id { get; set; }

        /// <summary>
        /// Name of the contact
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Email of the contact
        /// </summary>
        public string Email { get; set; }

        public EmailContact() {}

        public EmailContact(string name, string email, string id = null)
        {
            this.Id = id;
            this.Name = name;
            this.Email = email;
        }

        public EmailContact SetIdentifier(string id)
        {
            this.Id = id;
            return this;
        }

        public EmailContact SetName(string name)
        {
            this.Name = name;
            return this;
        }

        public EmailContact SetEmail(string email)
        {
            this.Email = email;
            return this;
        }
    }
}