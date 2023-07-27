#include <memory>

#include <chrono>
#include <cstdlib>
#include <memory>


#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"  //float64_multi_array   int16
#include "tm_msgs/srv/set_positions.hpp"
#include "tm_msgs/srv/send_script.hpp" //leave


using namespace std::chrono_literals;

float pos[7];


class testNode : public rclcpp::Node
{
private:
  float speed;
  float pos[7];
  /* data */
public:
  testNode(std::string name,float speed,float pos[7]):Node(name)
  {
    this->speed=speed;

    RCLCPP_INFO(this->get_logger(),"hellow%s\n%f",name.c_str(),this->speed); 
    for(int i=0;i<7;i++){
      this->pos[i]=pos[i];
      
    }


    

    move();
  }
  int move(){
    std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("demo_set_positions");

    rclcpp::Client<tm_msgs::srv::SetPositions>::SharedPtr client =
    node->create_client<tm_msgs::srv::SetPositions>("set_positions");
  
    
    if (pos[0]==0){
      auto request = std::make_shared<tm_msgs::srv::SetPositions::Request>();
      request->motion_type = tm_msgs::srv::SetPositions::Request::LINE_T;  //PTP_T  PTP_J
      for(int i=1;i<=6;i++){
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"num="<<this->pos[i]);
        request->positions.push_back(this->pos[i]);
      }
      request->velocity = this->speed;//rad/s
      request->acc_time = 0.2;
      request->blend_percentage = 10;
      request->fine_goal  = false;

      while (!client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
          RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
          return false;
        }
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
      }

      auto result = client->async_send_request(request);
      // Wait for the result.
      if (rclcpp::spin_until_future_complete(node, result) ==
        rclcpp::executor::FutureReturnCode::SUCCESS)
      {
        if(result.get()->ok){
          RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"OK");
        } else{
          RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"not OK");
        }

      } else {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Failed to call service");
      }
      return true;
    }else if(pos[0]==1){    
      auto request = std::make_shared<tm_msgs::srv::SetPositions::Request>();
      request->motion_type = tm_msgs::srv::SetPositions::Request::PTP_J;  //PTP_T  PTP_J
      for(int i=1;i<=6;i++){
        request->positions.push_back(this->pos[i]);
      }

      request->velocity = this->speed;//rad/s
      request->acc_time = 0.2;
      request->blend_percentage = 10;
      request->fine_goal  = false;

      while (!client->wait_for_service(1s)) {
        if (!rclcpp::ok()) {
          RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
          return false;
        }
        RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
      }

      auto result = client->async_send_request(request);
      // Wait for the result.
      if (rclcpp::spin_until_future_complete(node, result) ==
        rclcpp::executor::FutureReturnCode::SUCCESS)
      {
        if(result.get()->ok){
          RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"OK");
        } else{
          RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"not OK");
        }

      } else {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Failed to call service");
      }
      return true;
    }
    return true;
  }
};
bool send_cmd(std::string cmd, std::shared_ptr<rclcpp::Node> node, rclcpp::Client<tm_msgs::srv::SendScript>::SharedPtr client){
  auto request = std::make_shared<tm_msgs::srv::SendScript::Request>();
  request->id = "demo";
  request->script = cmd;

  while (!client->wait_for_service(1s)) {
    if (!rclcpp::ok()) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Interrupted while waiting for the service. Exiting.");
      return false;
    }
    RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"), "service not available, waiting again...");
  }

  auto result = client->async_send_request(request);
  // Wait for the result.
  if (rclcpp::spin_until_future_complete(node, result) ==
    rclcpp::executor::FutureReturnCode::SUCCESS)
  {
    if(result.get()->ok){
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"OK");
    } else{
      RCLCPP_INFO_STREAM(rclcpp::get_logger("rclcpp"),"not OK");
    }

  } else {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("rclcpp"), "Failed to call service");
  }
  return true;
}


class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(  //Float64MultiArray
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) const
  {
    RCLCPP_INFO(this->get_logger(), "I heard: '%f'", msg->data[1]);
    float speed=0.4;
    for(int i=0;i<7;i++){
      pos[i]=(float)msg->data[i];
    }

    auto node=std::make_shared<  testNode>("AA",speed,pos);
    std::shared_ptr<rclcpp::Node> node_leave = rclcpp::Node::make_shared("demo_leave_listen_node");
    rclcpp::Client<tm_msgs::srv::SendScript>::SharedPtr client =
      node_leave->create_client<tm_msgs::srv::SendScript>("send_script");
    std::string cmd = "ScriptExit()";
  
    send_cmd(cmd, node_leave, client);


    
  }
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());

  rclcpp::shutdown();


  return 0;
}