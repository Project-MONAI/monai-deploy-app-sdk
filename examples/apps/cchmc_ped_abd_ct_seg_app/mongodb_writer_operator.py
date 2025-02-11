# Copyright 2021-2025 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

from dotenv import load_dotenv

load_dotenv()

from pymongo import MongoClient, errors

from monai.deploy.core import Fragment, Operator, OperatorSpec


class MongoDBWriterOperator(Operator):
    """Class to write the MONAI Deploy Express MongoDB database with provided database entry.

    Named inputs:
        mongodb_database_entry: formatted MongoDB database entry.

    Named output:
        None

    Result:
        MONAI Deploy Express MongoDB database write of the database entry.
    """

    def __init__(self, fragment: Fragment, *args, database_name: str, collection_name: str, **kwargs):
        """Class to write the MONAI Deploy Express MongoDB database with provided database entry.

        Args:
            database_name (str): name of the MongoDB database that will be written.
            collection_name (str): name of the MongoDB collection that will be written.

        Raises:
            Relevant MongoDB errors if database writing is unsuccessful.
        """

        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))

        self.database_name = database_name
        self.collection_name = collection_name

        self.input_name_db_entry = "mongodb_database_entry"

        # MongoDB credentials
        self.mongodb_username = os.environ.get("MONGODB_USERNAME")
        self.mongodb_password = os.environ.get("MONGODB_PASSWORD")
        self.mongodb_port = os.environ.get("MONGODB_PORT")
        self.docker_mongodb_ip = os.environ.get("MONGODB_IP_DOCKER")

        # determine the MongoDB IP address based on execution environment
        self.mongo_ip = self._get_mongo_ip()
        self._logger.info(f"Using MongoDB IP: {self.mongo_ip}")

        # connect to the MongoDB database
        self.client = None

        try:
            self.client = MongoClient(
                f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongo_ip}:{self.mongodb_port}/?authSource=admin",
                serverSelectionTimeoutMS=10000,  # 10s timeout for testing connection; 20s by default
            )
            if self.client is None:
                raise RuntimeError("MongoClient was not created successfully")
            ping_response = self.client.admin.command("ping")
            self._logger.info(
                f"Successfully connected to MongoDB at: {self.client.address}. Ping response: {ping_response}"
            )
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
        except errors.ServerSelectionTimeoutError as e:
            self._logger.error("Failed to connect to MongoDB: Server selection timeout.")
            self._logger.debug(f"Detailed error: {e}")
            raise
        except errors.ConnectionFailure as e:
            self._logger.error("Failed to connect to MongoDB: Connection failure.")
            self._logger.debug(f"Detailed error: {e}")
            raise
        except errors.OperationFailure as e:
            self._logger.error("Failed to authenticate with MongoDB.")
            self._logger.debug(f"Detailed error: {e}")
            raise
        except Exception as e:
            self._logger.error("Unexpected error occurred while connecting to MongoDB.")
            self._logger.debug(f"Detailed error: {e}")
            raise
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """Set up the named input(s), and output(s) if applicable.

        This operator does not have an output for the next operator - MongoDB write only.

        Args:
            spec (OperatorSpec): The Operator specification for inputs and outputs etc.
        """

        spec.input(self.input_name_db_entry)

    def compute(self, op_input, op_output, context):
        """Performs computation for this operator"""

        mongodb_database_entry = op_input.receive(self.input_name_db_entry)

        # write to MongoDB
        self.write(mongodb_database_entry)

    def write(self, mongodb_database_entry):
        """Writes the database entry to the MONAI Deploy Express MongoDB database.

        Args:
            mongodb_database_entry: formatted MongoDB database entry.

        Returns:
            None
        """

        # MongoDB writing
        try:
            insert_result = self.collection.insert_one(mongodb_database_entry)
            if insert_result.acknowledged:
                self._logger.info(f"Document inserted with ID: {insert_result.inserted_id}")
            else:
                self._logger.error("Failed to write document to MongoDB.")
        except errors.PyMongoError as e:
            self._logger.error("Failed to insert document into MongoDB.")
            self._logger.debug(f"Detailed error: {e}")
            raise

    def _get_mongo_ip(self):
        """Determine the MongoDB IP based on the execution environment.

        If the pipeline is being run pythonically, use localhost.

        If MAP is being run via MAR or MONAI Deploy Express, use Docker bridge network IP.
        """

        # if running in a Docker container (/.dockerenv file present)
        if os.path.exists("/.dockerenv"):
            self._logger.info("Detected Docker environment")
            return self.docker_mongodb_ip

        # if not executing as Docker container, we are executing pythonically
        self._logger.info("Detected local environment (pythonic execution)")
        return "localhost"


# Module function (helper function)
def test():
    """Test writing to and deleting from the MDE MongoDB instance locally"""

    # MongoDB credentials
    mongodb_username = os.environ.get("MONGODB_USERNAME")
    mongodb_password = os.environ.get("MONGODB_PASSWORD")
    mongodb_port = os.environ.get("MONGODB_PORT")

    # sample information
    database_name = "CTLiverSpleenSegPredictions"
    collection_name = "OrganVolumes"
    test_entry = {"test_key": "test_value"}

    # connect to MongoDB instance (localhost as we are testing locally)
    try:
        client = MongoClient(
            f"mongodb://{mongodb_username}:{mongodb_password}@localhost:{mongodb_port}/?authSource=admin",
            serverSelectionTimeoutMS=10000,  # 10s timeout for testing connection; 20s by default
        )
        if client is None:
            raise RuntimeError("MongoClient was not created successfully")
        ping_response = client.admin.command("ping")
        print(f"Successfully connected to MongoDB at: {client.address}. Ping response: {ping_response}")
        db = client[database_name]
        collection = db[collection_name]
    except errors.ServerSelectionTimeoutError as e:
        print("Failed to connect to MongoDB: Server selection timeout.")
        print(f"Detailed error: {e}")
        raise
    except errors.ConnectionFailure as e:
        print("Failed to connect to MongoDB: Connection failure.")
        print(f"Detailed error: {e}")
        raise
    except errors.OperationFailure as e:
        print("Failed to authenticate with MongoDB.")
        print(f"Detailed error: {e}")
        raise
    except Exception as e:
        print("Unexpected error occurred while connecting to MongoDB.")
        print(f"Detailed error: {e}")
        raise

    # insert document
    try:
        insert_result = collection.insert_one(test_entry)
        if insert_result.acknowledged:
            print(f"Document inserted with ID: {insert_result.inserted_id}")
        else:
            print("Failed to write document to MongoDB.")
    except errors.PyMongoError as e:
        print("Failed to insert document into MongoDB.")
        print(f"Detailed error: {e}")
        raise

    # verify the inserted document
    try:
        inserted_doc = collection.find_one({"_id": insert_result.inserted_id})
        if inserted_doc:
            print(f"Inserted document: {inserted_doc}")
        else:
            print("Document not found in the collection after insertion.")
    except errors.PyMongoError as e:
        print("Failed to retrieve the inserted document from MongoDB.")
        print(f"Detailed error: {e}")
        return

    # # delete a database
    # try:
    #     client.drop_database(database_name)
    #     print(f"Test database '{database_name}' deleted successfully.")
    # except errors.PyMongoError as e:
    #     print("Failed to delete the test database.")
    #     print(f"Detailed error: {e}")


if __name__ == "__main__":
    test()
