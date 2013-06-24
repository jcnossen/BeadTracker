using System;
using System.Collections.Generic;
using System.Text;

using Comfirm.AlphaMail.Services.Client;
using Comfirm.AlphaMail.Services.Client.Entities;
using Comfirm.AlphaMail.Services.Client.Exceptions;


namespace BeadTrackerEmailNotifier
{
    public class BeadTrackerEmailNotifier
    {
        string apiToken;

        public string ApiToken
        {
            get { return apiToken; }
            set { apiToken = value; }
        }
        int projectId;

        public int ProjectId
        {
            get { return projectId; }
            set { projectId = value; }
        }

        public BeadTrackerEmailNotifier()
        {}

        public BeadTrackerEmailNotifier(string apiToken, int projectId)
        {
            this.apiToken = apiToken;
            this.projectId = projectId;
        }

        public void SendEmail(string senderEmail, string senderName, string receiverEmail, string receiverName,
            string subject, string content)
        {
            var emailService = new AlphaMailEmailService(apiToken);

            var message = new
            {
                subject = subject,
                content = content
            };

            var payload = new EmailMessagePayload(projectId, 0, new EmailContact(receiverName, receiverEmail),
                new EmailContact(senderName, senderEmail), message);

            try
            {
                var response = emailService.Queue(payload);
                Console.WriteLine("Mail successfully sent! ID = {0}", response.Result);
            }
            catch (AlphaMailServiceException exception)
            {
                Console.WriteLine("Error! {0} ({1})", exception.Message, exception.GetErrorCode());
            }
        }
    }
}
