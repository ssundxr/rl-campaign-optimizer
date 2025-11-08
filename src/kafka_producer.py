"""
Kafka Event Producer
Simulates real-time customer events (clicks, purchases, churns)
"""

from kafka import KafkaProducer
import json
import time
import random
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerEventProducer:
    """Produces customer events to Kafka topic"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='customer_events'):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        logger.info(f"Kafka Producer initialized for topic: {topic}")
    
    def generate_event(self):
        """Generate synthetic customer event"""
        event_types = ['click', 'purchase', 'churn', 'email_open']
        
        event = {
            'customer_id': random.randint(1000, 9999),
            'event_type': random.choice(event_types),
            'timestamp': datetime.now().isoformat(),
            'campaign_id': random.randint(1, 10),
            'value': random.uniform(0, 500)
        }
        
        return event
    
    def send_events(self, num_events=100, delay=1.0):
        """Send events to Kafka topic"""
        logger.info(f"Starting to send {num_events} events...")
        
        for i in range(num_events):
            event = self.generate_event()
            self.producer.send(self.topic, value=event)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Sent {i + 1} events")
            
            time.sleep(delay)
        
        self.producer.flush()
        logger.info("All events sent successfully")
    
    def close(self):
        """Close Kafka producer"""
        self.producer.close()
        logger.info("Kafka Producer closed")


def main():
    """Main execution"""
    producer = CustomerEventProducer()
    
    try:
        producer.send_events(num_events=100, delay=0.5)
    finally:
        producer.close()


if __name__ == "__main__":
    main()
